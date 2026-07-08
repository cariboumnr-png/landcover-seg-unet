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
Top-level namespace for `geopipe.foundation.data_blocks.assembler`.

This submodule coordinates the preparation, construction, and
structural validation of individual `.npz` block files. It decouples
core domain block representations from raw raster reading (I/O) and
coordinates the parallelized assembly of blocks over mapped raster
windows.

Public APIs:
    - BlockBuilderConfig: Config options for lifecycle pipeline.
    - BlockBuilderResult: Dataclass result wrapping builder execution outputs.
    - BlockCreationContext: Source parameters for single blocks.
    - RasterReadInput: Dataclass specs parameter for reading rasters.
    - RasterReadOutput: Dataclass container for read raster arrays.
    - build_single_block: Constructs a block from windowed rasters.
    - build_test_block: Finds, normalizes, and saves a test block.
    - build_blocks: Parallelized multiblock checking and assembly.
    - read_block_raster_data: Reads image/label and DEM bands.
    - check_npz_integrity: Verifies a saved .npz file is readable.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuilderConfig',
    'BlockBuilderResult',
    'BlockCreationContext',
    'RasterReadInput',
    'RasterReadOutput',
    # functions
    'build_single_block',
    'build_test_block',
    'build_blocks',
    'read_block_raster_data',
    'check_npz_integrity',
]

if typing.TYPE_CHECKING:
    from .lifecycle import BlockBuilderConfig, BlockBuilderResult, build_blocks
    from .assembler import (
        BlockCreationContext,
        build_single_block,
        build_test_block,
    )
    from .io import (
        RasterReadInput,
        RasterReadOutput,
        read_block_raster_data,
        check_npz_integrity,
    )


def __getattr__(name: str):
    if name in {'BlockBuilderConfig', 'BlockBuilderResult', 'build_blocks'}:
        return getattr(
            importlib.import_module('.lifecycle', __package__), name
        )

    if name in {
        'BlockCreationContext',
        'build_single_block',
        'build_test_block',
    }:
        return getattr(
            importlib.import_module('.assembler', __package__), name
        )

    if name in {
        'RasterReadInput',
        'RasterReadOutput',
        'read_block_raster_data',
        'check_npz_integrity',
    }:
        return getattr(
            importlib.import_module('.io', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
