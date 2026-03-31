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
Type aliases for `geopipe.foundation` for raster I/O and grid windows.

Centralizes third-party types from `rasterio` and `numpy.typing` so call sites
have concise, consistent annotations.
'''

# standard imports
import typing
# third-party imports
import numpy.typing
import rasterio.io
import rasterio.windows

# rasterio types
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
'''
A mapping of pixel-origin coordinates `(x_px, y_px)` to
`rasterio.windows.Window` objects from the world grid.
'''

RasterWindow: typing.TypeAlias = rasterio.windows.Window
'''
Array read from a raster window with its top-left corner at the given
pixel-origin coordinates `(x_px, y_px)` in world-grid space.
'''

RasterWindowDict: typing.TypeAlias = dict[tuple[int, int], RasterWindow]
'''
A collection of `rasterio.windows.Window` indexed by coordinates  (x, y
in pixels) from the world grid.
'''

RasterTile: typing.TypeAlias = tuple[tuple[int, int], numpy.typing.NDArray]
'''
Array read from a raster window with its top-left corner at specified
coordinates (x, y in pixels) from the world grid.
'''

RasterTileDict: typing.TypeAlias = dict[tuple[int, int], numpy.typing.NDArray]
'''
A mapping from pixel-origin coordinates `(x_px, y_px)` to NumPy arrays
holding the tile data read from corresponding raster windows.
'''

RasterTransform: typing.TypeAlias = rasterio.Affine | None
'''
Affine transform for a raster; `None` when a transform is unavailable.
'''
