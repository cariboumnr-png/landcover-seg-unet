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
class BlockBuildConfig:
    '''
    Configuration parameters for building a data block from source.
    '''
    img_path: str
    lbl_path: str | None
    image_band_map: dict[str, int]
    label_specs: dict[str, geo_core.LabelSpecs] | None = None
    dem_pad_px: int = 0
    add_spectral: list[str] | None = None
    add_topo: bool = False


@dataclasses.dataclass(frozen=True)
class BlockCreationContext:
    '''
    Spatial coordinate contexts for a single block build.
    '''
    name: str
    img_window: alias.RasterWindow
    lbl_window: alias.RasterWindow | None


@dataclasses.dataclass(frozen=True)
class TestBlockJob:
    '''
    Execution job parameters for test block search.
    '''
    config: BlockBuildConfig
    save_dpath: str
    valid_px_per: float
    monitor_head: str
    need_all_classes: bool
    logger: utils.Logger
    ignore_index: int = 255


def build_single_block(
    context: BlockCreationContext,
    config: BlockBuildConfig,
    ignore_index: int = 255,
    *,
    save: bool = False,
    save_fpath: str | None = None
) -> geo_core.DataBlock:
    '''
    Create a DataBlock from input rasters for the window context.

    Args:
        context: Context object defining spatial window.
        config: Configuration parameters for path and features.
        ignore_index: Label ignore value.
        save: If True, compresses and writes the block to disk.
        save_fpath: Target destination path for saving the block.

    Returns:
        DataBlock: A populated and validated block instance.
    '''
    read_inputs = assembler.RasterReadInput(
        image_fpath=config.img_path,
        label_fpath=config.lbl_path,
        image_window=context.img_window,
        label_window=context.lbl_window,
        dem_pad_px=config.dem_pad_px,
        image_band_map=config.image_band_map,
        label_specs=config.label_specs
    )
    read_outputs = assembler.read_block_raster_data(read_inputs)

    has_lbl = read_outputs.label_array is not None

    core_config = geo_core.DataBlockConfig(
        image_band_map=config.image_band_map,
        ignore_index=ignore_index,
        dem_pad_px=config.dem_pad_px,
        image_nodata=read_outputs.image_nodata,
        label_nodata=read_outputs.label_nodata,
        add_spectral=config.add_spectral,
        add_topo=config.add_topo
    )

    build_ctx = geo_core.DataBlockBuildContext(
        block_name=context.name,
        image_array=read_outputs.image_array,
        image_padded_dem=(
            read_outputs.padded_dem if config.add_topo else None
        ),
        label_array=read_outputs.label_array,
        label_specs=config.label_specs if has_lbl else None,
        config=core_config
    )

    block = geo_core.DataBlock.build(build_ctx)

    if save:
        assert save_fpath
        block.save(save_fpath)
    return block


def build_test_block(
    contexts: list[BlockCreationContext],
    job: TestBlockJob
) -> str | None:
    '''
    Build, normalize, and persist a single valid block for testing.

    Iterates deterministically over available block contexts to find
    the first block satisfying the validity and label-coverage
    criteria. Normalizes the block using local mean/std before saving.

    Args:
        contexts: List of candidate block creation contexts.
        job: Execution parameters including config, criteria, and logger.

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
            temp_blk = build_single_block(
                ctx, job.config, job.ignore_index, save=False
            )
            blk_meta = temp_blk.meta

            # Check valid pixel ratios based on image band (fix KeyError)
            ratios = blk_meta['valid_ratios']
            block_ratio_ok = ratios.get('image', 0.0) >= job.valid_px_per

            has_all_labels = True
            if job.monitor_head in blk_meta['label_count']:
                has_all_labels = all(
                    blk_meta['label_count'][job.monitor_head]
                )
            else:
                has_all_labels = False

            if block_ratio_ok and (
                has_all_labels or not job.need_all_classes
            ):
                blk = temp_blk
                chosen_context = ctx
                break
        except ValueError:
            continue

    if not blk:
        job.logger.log('WARNING', 'No valid block for testing.')
        return None

    # In-place image normalization for debugging block
    mean = numpy.mean(blk.data.image)
    std = numpy.std(blk.data.image)
    blk.data.image = (blk.data.image - mean) / (std or 1.0)

    msg = f'Fetched a valid block at context: {chosen_context.name}'
    job.logger.log('INFO', msg)
    job.logger.log('DEBUG', 'Criteria:')
    job.logger.log('DEBUG', f'Minimum valid pixel: {job.valid_px_per:.2f}')
    job.logger.log('DEBUG', f'Focused head: {job.monitor_head}')
    job.logger.log('DEBUG', f'Requires all classes: {job.need_all_classes}')

    os.makedirs(job.save_dpath, exist_ok=True)
    fpath = os.path.join(job.save_dpath, f'{chosen_context.name}.npz')
    blk.save(fpath)
    return fpath
