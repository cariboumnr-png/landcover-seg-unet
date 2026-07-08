# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Domain logic for building data blocks and managing test blocks.

This module encapsulates block-level feature engineering and array
construction rules. It manages coordinate context mappings, calculates
derived spectral indices, validates spatial alignments, and builds
fully materialized `DataBlock` instances. Additionally, it handles
test-block extraction with in-place normalization and criteria
checking.

Public APIs:
    - BlockCreationContext: Parameter object holding block read specs.
    - build_single_block: Constructs a block from input rasters.
    - build_test_block: Finds, normalizes, and saves a test block.
'''

# standard imports
import copy
import dataclasses
import os
import random
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.assembler as assembler
import landseg.utils as utils


@dataclasses.dataclass(frozen=True)
class BlockCreationContext:
    '''Inputs required to build a single data block from source rasters.'''
    name: str
    dem_pad_px: int
    img_path: str
    img_window: alias.RasterWindow
    lbl_path: str | None
    lbl_window: alias.RasterWindow | None
    label_specs: dict[str, geo_core.LabelSpecs] | None


def build_single_block(
    meta: geo_core.DataBlockMeta,
    context: BlockCreationContext,
    *,
    save: bool = False,
    save_fpath: str | None = None
) -> geo_core.DataBlock:
    '''
    Create a DataBlock from input rasters for the window context.

    Args:
        meta: Canonical block metadata template.
        context: Context object defining paths, windows, and specs.
        save: If True, compresses and writes the block to disk.
        save_fpath: Target destination path for saving the block.

    Returns:
        DataBlock: A populated and validated block instance.
    '''
    meta_copy = copy.deepcopy(meta)
    meta_copy['block_name'] = context.name

    read_inputs = assembler.RasterReadInput(
        image_fpath=context.img_path,
        label_fpath=context.lbl_path,
        image_window=context.img_window,
        label_window=context.lbl_window,
        dem_pad_px=context.dem_pad_px,
        image_band_map=meta_copy['image_band_map'],
        label_specs=context.label_specs
    )
    read_outputs = assembler.read_block_raster_data(read_inputs)

    meta_copy['image_nodata'] = read_outputs.image_nodata
    meta_copy['image_dem_pad'] = context.dem_pad_px
    if read_outputs.label_nodata is not None:
        meta_copy['label_nodata'] = read_outputs.label_nodata

    # Determine which spectral indices to add based on band availability
    spectral_indices = []
    band_names = {b.lower() for b in meta_copy['image_band_map']}
    if 'red' in band_names:
        if 'nir' in band_names:
            spectral_indices.append('ndvi')
            if 'swir1' in band_names:
                spectral_indices.append('ndmi')
            if 'swir2' in band_names:
                spectral_indices.append('nbr')

    has_dem = 'dem' in band_names
    has_lbl = read_outputs.label_array is not None

    build_ctx = geo_core.DataBlockBuildContext(
        block_meta=meta_copy,
        image_array=read_outputs.image_array,
        image_padded_dem=read_outputs.padded_dem if has_dem else None,
        image_add_spectral=spectral_indices,
        image_add_topo=has_dem,
        label_array=read_outputs.label_array,
        label_specs=context.label_specs if has_lbl else None
    )

    block = geo_core.DataBlock.build(build_ctx)

    if save:
        assert save_fpath
        block.save(save_fpath)
    return block


def build_test_block(
    meta: geo_core.DataBlockMeta,
    contexts: list[BlockCreationContext],
    save_dpath: str,
    *,
    valid_px_per: float,
    monitor_head: str,
    need_all_classes: bool,
    logger: utils.Logger
) -> str | None:
    '''
    Build, normalize, and persist a single valid block for testing.

    Iterates deterministically over available block contexts to find
    the first block satisfying the validity and label-coverage
    criteria. Normalizes the block using local mean/std before saving.

    Args:
        meta: Core metadata dictionary.
        contexts: List of candidate block creation contexts.
        save_dpath: Target folder where the block will be saved.
        valid_px_per: Minimum fraction of valid pixels required.
        monitor_head: Label head used to monitor class coverage.
        need_all_classes: If True, require all classes to be present.
        logger: Diagnostic progress logger.

    Returns:
        str | None: The saved block path if found, otherwise None.
    '''
    shuffled_contexts = list(contexts)
    random.Random(42).shuffle(shuffled_contexts)

    blk: geo_core.DataBlock | None = None
    chosen_context: BlockCreationContext | None = None
    for ctx in shuffled_contexts:
        print('Searching for a valid block...', end='\r', flush=True)
        try:
            temp_blk = build_single_block(meta, ctx, save=False)
            blk_meta = temp_blk.meta

            # Check valid pixel ratios based on image band (fix KeyError)
            ratios = blk_meta['valid_ratios']
            block_ratio_ok = ratios.get('image', 0.0) >= valid_px_per

            has_all_labels = True
            if monitor_head in blk_meta['label_count']:
                has_all_labels = all(blk_meta['label_count'][monitor_head])
            else:
                has_all_labels = False

            if block_ratio_ok and (has_all_labels or not need_all_classes):
                blk = temp_blk
                chosen_context = ctx
                break
        except ValueError:
            continue

    if not blk:
        logger.log('WARNING', 'No valid block for testing.')
        return None

    # In-place image normalization for debugging block
    mean = numpy.mean(blk.data.image)
    std = numpy.std(blk.data.image)
    blk.data.image = (blk.data.image - mean) / (std or 1.0)

    msg = f'Fetched a valid block at context: {chosen_context.name}'
    logger.log('INFO', msg)
    logger.log('DEBUG', 'Criteria:')
    logger.log('DEBUG', f'Minimum valid pixel: {valid_px_per:.2f}')
    logger.log('DEBUG', f'Focused head: {monitor_head}')
    logger.log('DEBUG', f'Requires all classes: {need_all_classes}')

    os.makedirs(save_dpath, exist_ok=True)
    fpath = os.path.join(save_dpath, f'{chosen_context.name}.npz')
    blk.save(fpath)
    return fpath
