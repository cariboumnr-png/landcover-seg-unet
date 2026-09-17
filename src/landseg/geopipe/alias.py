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
Foundational typing aliases for the `landseg.geopipe` package.

This module provides lightweight, dependency-free base type aliases
shared across the entire geospatial pipeline, including 2D integer
coordinate primitives, raster I/O handles, typed NumPy arrays, and
common class frequency mappings.

Public APIs:
    - `Coord2d`: 2D integer coordinate pair (x, y) or (row, col).
    - `CoordsList`: List of 2D integer coordinate pairs.
    - `CoordsSet`: Set of 2D integer coordinate pairs.
    - `RasterReader`: Rasterio DatasetReader handle.
    - `RasterWindow`: Rasterio Window slice.
    - `RasterWindowDict`: Mapping of coordinates to raster windows.
    - `RasterTransform`: Rasterio affine transform or None.
    - `IntArray`: Generic NumPy integer array.
    - `Int64Array`: NumPy array with int64 dtype.
    - `Float32Array`: NumPy array with float32 dtype.
    - `Float64Array`: NumPy array with float64 dtype.
    - `MaskArray`: NumPy boolean mask array.
    - `ClassCounts`: Mapping of head names to class count lists.
    - `CoordClassCounts`: Mapping of coordinates to class count lists.
'''

# standard imports
from __future__ import annotations
import typing
# third-party imports
import numpy.typing
import rasterio
import rasterio.io
import rasterio.windows


# ----- typing aliases: spatial coordinates
Coord2d: typing.TypeAlias = tuple[int, int]
'''Two-dimensional integer coordinate pair (x, y) or (row, col).'''

CoordsList: typing.TypeAlias = list[Coord2d]
'''Ordered list of two-dimensional integer coordinate pairs.'''

CoordsSet: typing.TypeAlias = set[Coord2d]
'''Unordered set of two-dimensional integer coordinate pairs.'''


# ----- typing aliases: raster i/o and geometry
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
'''Dataset reader handle for opened raster files.'''

RasterWindow: typing.TypeAlias = rasterio.windows.Window
'''Pixel window offset and slice for raster reads and writes.'''

RasterWindowDict: typing.TypeAlias = dict[Coord2d, RasterWindow]
'''Mapping of pixel-origin coordinates to raster windows.'''

RasterTransform: typing.TypeAlias = rasterio.Affine | None
'''Affine transform matrix or None when unavailable.'''


# ----- typing aliases: numpy arrays
IntArray: typing.TypeAlias = numpy.typing.NDArray[numpy.integer]
'''Generic NumPy integer array of arbitrary bit width.'''

Int64Array: typing.TypeAlias = numpy.typing.NDArray[numpy.int64]
'''NumPy array with int64 dtype for indices and labels.'''

Float32Array: typing.TypeAlias = numpy.typing.NDArray[numpy.float32]
'''NumPy array with float32 dtype for imagery and features.'''

Float64Array: typing.TypeAlias = numpy.typing.NDArray[numpy.float64]
'''NumPy array with float64 dtype for continuous domain maps.'''

MaskArray: typing.TypeAlias = numpy.typing.NDArray[numpy.bool]
'''NumPy boolean mask array for valid-pixel filters.'''


# ----- typing aliases: dataset and class statistics
ClassCounts: typing.TypeAlias = dict[str, list[int]]
'''Mapping of head or dataset names to class frequency lists.'''

CoordClassCounts: typing.TypeAlias = dict[Coord2d, list[int]]
'''Mapping of block coordinates to class frequency lists.'''
