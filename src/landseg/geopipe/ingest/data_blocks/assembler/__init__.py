# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Top-level namespace for `geopipe.ingest.data_blocks.assembler`.

This submodule coordinates the preparation, construction, and
structural validation of individual `.npz` block files. It decouples
core domain block representations from raw raster reading (I/O) and
coordinates the parallelized assembly of blocks over mapped raster
windows.

Public APIs:
    - BlockBuildingInput: I/O path options for lifecycle pipeline.
    - BlockBuildingContext: Read windows for pipeline execution.
    - BlockBuildingConfig: Container for building configurations.
    - BlockBuildingOutput: Result wrapping builder execution outputs.
    - RasterReadInput: Specs parameter for reading raster inputs.
    - RasterReadOutput: Container holding read raster numpy arrays.
    - DataBlockInputs: container for raw arrays to construct a DataBlock.
    - DataBlockConfig: build-time configuration for feature engineering.
    - build_data_block: construct a DataBlock with derived features.
    - build_single_block: Constructs a block from windowed rasters.
    - build_test_block: Finds, normalizes, and saves a test block.
    - build_blocks: Parallelized multiblock checking and assembly.
    - read_band_map: Reads and validates raster band mapping dict.
    - read_label_specs: Reads and formats label specification dict.
    - read_schemes: Reads and maps categorical scheme definitions.
    - read_block_raster_data: Reads image/label and DEM bands.
    - check_npz_integrity: Verifies a saved .npz file is readable.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuildingInput',
    'BlockBuildingContext',
    'BlockBuildingConfig',
    'BlockBuildingOutput',
    'RasterReadInput',
    # functions
    'build_blocks',
    'build_test_block',
    'read_band_map',
    'read_label_specs',
    'read_schemes',
]

if typing.TYPE_CHECKING:
    from .lifecycle import (
        BlockBuildingInput,
        BlockBuildingContext,
        BlockBuildingConfig,
        BlockBuildingOutput,
        build_blocks,
        build_test_block,
    )


    from .io import (
        RasterReadInput,
        read_band_map,
        read_label_specs,
        read_schemes,
    )


def __getattr__(name: str):
    if name in {
        'BlockBuildingInput',
        'BlockBuildingContext',
        'BlockBuildingConfig',
        'BlockBuildingOutput',
        'build_blocks',
        'build_test_block',
    }:
        return getattr(
            importlib.import_module('.lifecycle', __package__), name
        )

    if name in {
        'RasterReadInput',
        'read_band_map',
        'read_label_specs',
        'read_schemes',
    }:
        return getattr(
            importlib.import_module('.io', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
