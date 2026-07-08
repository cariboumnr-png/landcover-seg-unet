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
    - build_single_block: Constructs a block from input rasters.
    - build_test_block: Finds, normalizes, and saves a test block.
'''

# standard imports
import os
import random
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks.assembler as assembler


def build_single_block(
    name: str,
    inputs: assembler.RasterReadInput,
    *,
    ignore_index: int = 255,
    add_spectral: list[str] | None = None,
    add_topo: bool = False,
    save_fpath: str | None = None,
) -> geo_core.DataBlock:
    '''
    Create a DataBlock from input rasters for the window context.

    Args:
        name:
            Unique identifier for the block.
        inputs:
            Raster inputs and metadata required to construct the block.
        save_fpath:
            Optional output path. If provided, the constructed block is
            serialized to this location.
        ignore_index:
            Label value assigned to ignored pixels in the output block.
        add_spectral:
            Optional list of spectral indices to compute and append as
            image bands.
        add_topo:
            Whether to compute and append topographic features derived
            from the DEM.

    Returns:
        DataBlock: A populated and validated block instance.
    '''
    read_outputs = assembler.read_block_raster_data(inputs)

    datablock_inputs = geo_core.DataBlockInputs(
        block_name=name,
        image_array=read_outputs.image_array,
        image_padded_dem=read_outputs.image_padded_dem,
        label_array=read_outputs.label_array,
        label_specs=inputs.label_specs,
    )
    datablock_config = geo_core.DataBlockConfig(
        image_band_map=inputs.image_band_map,
        image_dem_pad_px=inputs.image_dem_pad_px,
        image_nodata=read_outputs.image_nodata,
        label_nodata=read_outputs.label_nodata,

        label_ignore_index=ignore_index,
        add_spectral=add_spectral,
        add_topo=add_topo
    )
    block = geo_core.DataBlock.build(datablock_inputs, datablock_config)

    if save_fpath:
        block.save(save_fpath)
    return block


def build_test_block(
    save_dpath: str,
    inputs: dict[str, assembler.RasterReadInput],
    *,
    target_head: str,
    valid_px_per: float,
    need_all_classes: bool,
) -> str | None:
    '''
    Build, normalize, and persist a single valid block for testing.

    Iterates over the available block inputs in a deterministic shuffled
    order to find the first block meeting the validity and label-coverage
    criteria. The selected block is normalized using its own image mean
    and standard deviation before being saved.

    Args:
        save_dpath:
            Directory where the test block will be written.
        inputs:
            Mapping from block name to raster inputs.
        target_head:
            Label head used when checking class coverage.
        valid_px_per:
            Minimum required proportion of valid image pixels.
        need_all_classes:
            Whether every class in the target head must be present for a
            block to be accepted.

    Returns:
        str | None: Path to the saved test block if one is found;
            otherwise, ``None``.
    '''
    shuffled_inputs = list(inputs.items())
    random.Random(42).shuffle(shuffled_inputs)

    name: str | None = None
    candidate: geo_core.DataBlock | None = None
    for name, candidate_input in shuffled_inputs:
        print('Searching for a valid block...', end='\r', flush=True)
        try:
            candidate = build_single_block(name, candidate_input)
            manifest = candidate.manifest

            # check valid pixel ratios based on image band
            valid_ratio = manifest['valid_ratios'].get('image', 0.0)

            # check if all classes are present
            has_all_classes = True
            if target_head in manifest['label_count']:
                has_all_classes = all(manifest['label_count'][target_head])
            else:
                has_all_classes = False

            if (
                valid_ratio >= valid_px_per and
                (has_all_classes or not need_all_classes)
            ):
                break
        except ValueError:
            continue

    if not candidate:
        return None

    # In-place image normalization for debugging block
    mean = numpy.mean(candidate.data.image)
    std = numpy.std(candidate.data.image)
    candidate.data.image = (candidate.data.image - mean) / (std or 1.0)

    os.makedirs(save_dpath, exist_ok=True)
    fpath = os.path.join(save_dpath, f'test_{name}.npz')
    candidate.save(fpath)
    return fpath
