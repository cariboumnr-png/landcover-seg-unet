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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
'''GridSpec protocol.'''

# standard imports
import collections.abc
import typing
# local imports
import landseg.geopipe.core.alias as alias

# -------------------------------Public Protocol-------------------------------
@typing.runtime_checkable
class GridLayoutLike(typing.Protocol):
    '''Used by ingestion modules.'''
    def keys(self) -> collections.abc.KeysView[tuple[int, int]]: ...
    def items(self) -> collections.abc.ItemsView[tuple[int, int], alias.RasterWindow]: ...
    def offset_from(self, src) -> None:...
    @property
    def gid(self) -> str:...
    @property
    def crs(self) -> str:...
    @property
    def origin(self) -> tuple[float, float]:...
    @property
    def tile_size(self) -> tuple[int, int]:...
    @property
    def tile_overlap(self) -> tuple[int, int]:...

class MappedRasterWindowsLike(typing.Protocol):
    '''From raster alignment to the grid.'''
    @property
    def grid_id(self) -> str:...
    @property
    def tile_shape(self) -> tuple[int, int]:...
    @property
    def image(self) -> alias.RasterWindowDict:...
    @property
    def label(self) -> alias.RasterWindowDict:...
