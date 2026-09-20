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
Top-level namespace for `landseg.geopipe.ingest.blocks.assembler`.

This submodule coordinates the preparation, construction, and
structural validation of individual `.npz` block files. It decouples
core domain block representations from raw raster reading (I/O) and
coordinates the parallelized assembly of blocks over mapped raster
windows.

Public APIs:
    - `BlockBuildingConfig`: Container for building configurations.
    - `BlockBuildingContext`: Read windows for pipeline execution.
    - `BlockBuildingInput`: I/O path options for lifecycle pipeline.
    - `BlockBuildingOutput`: Result wrapping builder execution outputs.
    - `DataBlockConfig`: Configuration for feature engineering.
    - `DataBlockInputs`: Container for raw array data block inputs.
    - `RasterReadInput`: Specs parameter for reading raster inputs.
    - `build_blocks`: Parallelized multiblock checking and assembly.
    - `build_data_block`: Construct a DataBlock with derived features.
    - `build_test_block`: Finds, normalizes, and saves a test block.
    - `read_band_map`: Reads and validates raster band mapping dict.
    - `read_label_specs`: Reads and formats label specification dict.
    - `read_schemes`: Reads and maps categorical scheme definitions.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuildingConfig',
    'BlockBuildingContext',
    'BlockBuildingInput',
    'BlockBuildingOutput',
    'DataBlockConfig',
    'DataBlockInputs',
    'RasterReadInput',
    # functions
    'build_blocks',
    'build_data_block',
    'build_test_block',
    'read_band_map',
    'read_label_specs',
    'read_schemes',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import (
        DataBlockConfig,
        DataBlockInputs,
        build_data_block,
    )
    from .io import (
        RasterReadInput,
        read_band_map,
        read_label_specs,
        read_schemes,
    )
    from .lifecycle import (
        BlockBuildingConfig,
        BlockBuildingContext,
        BlockBuildingInput,
        BlockBuildingOutput,
        build_blocks,
        build_test_block,
    )


def __getattr__(name: str):
    if name in {
        'DataBlockConfig',
        'DataBlockInputs',
        'build_data_block',
    }:
        obj = importlib.import_module('.builder', __package__)
        return getattr(obj, name)

    if name in {
        'RasterReadInput',
        'read_band_map',
        'read_label_specs',
        'read_schemes',
    }:
        obj = importlib.import_module('.io', __package__)
        return getattr(obj, name)

    if name in {
        'BlockBuildingConfig',
        'BlockBuildingContext',
        'BlockBuildingInput',
        'BlockBuildingOutput',
        'build_blocks',
        'build_test_block',
    }:
        obj = importlib.import_module('.lifecycle', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
