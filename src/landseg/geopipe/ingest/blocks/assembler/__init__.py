# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
    - `BlockBuildingInputs`: Source raster paths and window mappings.
    - `BlockBuildingOutput`: Results and summary statistics.
    - `RasterReadContext`: Immutable raster context and metadata.
    - `RasterReadOutput`: Read arrays, nodata values, and metadata.
    - `build_blocks`: Parallelized multiblock checking and assembly.
    - `build_data_block`: Construct a DataBlock with derived features.
    - `read_block_raster_data`: Reads windowed raster arrays and DEM.
    - `read_schemes`: Reads and maps categorical scheme definitions.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'BlockBuildingConfig',
    'BlockBuildingInputs',
    'BlockBuildingOutput',
    'RasterReadContext',
    'RasterReadOutput',
    'build_blocks',
    'build_data_block',
    'read_block_raster_data',
    'read_schemes',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import build_data_block
    from .io import (
        RasterReadContext,
        RasterReadOutput,
        read_block_raster_data,
        read_schemes,
    )
    from .lifecycle import (
        BlockBuildingConfig,
        BlockBuildingInputs,
        BlockBuildingOutput,
        build_blocks,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.builder': ['build_data_block'],
    '.io': [
        'RasterReadContext',
        'RasterReadOutput',
        'read_block_raster_data',
        'read_schemes',
    ],
    '.lifecycle': [
        'BlockBuildingConfig',
        'BlockBuildingInputs',
        'BlockBuildingOutput',
        'build_blocks',
    ],
})
