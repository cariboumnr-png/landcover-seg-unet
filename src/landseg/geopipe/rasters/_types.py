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

'''`rasterio` types and aliases.'''

# standard imports
import typing
# third-party imports
import rasterio
import rasterio.coords
import rasterio.io
import rasterio.transform
import rasterio.windows


# ----- typing aliases
Affine: typing.TypeAlias = rasterio.transform.Affine
'''Raster transform affine.'''

BoundingBox: typing.TypeAlias = rasterio.coords.BoundingBox
'''Raster bound box.'''

DatasetReader: typing.TypeAlias = rasterio.io.DatasetReader
'''Dataset reader handle for opened raster files.'''

Window: typing.TypeAlias = rasterio.windows.Window
'''Pixel window offset and slice for raster reads and writes.'''
