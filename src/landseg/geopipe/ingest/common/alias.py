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
Type aliases for raster I/O, windows, and tiles in data ingestion.

This module defines common type aliases for rasterio reader handles,
window mappings, and array tile dictionaries.

Public APIs:
    - `RasterReader`: Type alias for rasterio DatasetReader.
    - `RasterWindow`: Type alias for rasterio Window.
    - `RasterWindowDict`: Type alias for coordinate to Window mapping.
    - `RasterTile`: Type alias for coordinate and array tuple.
    - `RasterTileDict`: Type alias for coordinate to array mapping.
    - `RasterTransform`: Type alias for raster affine transform or None.
'''

# standard imports
from __future__ import annotations
import typing
# third-party imports
import numpy.typing
import rasterio.io
import rasterio.windows


# ----- typing aliases
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
'''Type alias for rasterio DatasetReader.'''

RasterWindow: typing.TypeAlias = rasterio.windows.Window
'''Type alias for rasterio Window.'''

RasterWindowDict: typing.TypeAlias = dict[tuple[int, int], RasterWindow]
'''Mapping of pixel-origin coordinates (x, y) to raster windows.'''

RasterTile: typing.TypeAlias = tuple[tuple[int, int], numpy.typing.NDArray]
'''Tuple of pixel coordinates (x, y) and tile array data.'''

RasterTileDict: typing.TypeAlias = dict[tuple[int, int], numpy.typing.NDArray]
'''Mapping of pixel coordinates (x, y) to tile array data.'''

RasterTransform: typing.TypeAlias = rasterio.Affine | None
'''Affine transform for a raster or None when unavailable.'''
