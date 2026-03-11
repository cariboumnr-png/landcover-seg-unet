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
# pylint: disable=too-few-public-methods
'''`DataBlock` protocol.'''

# standard imports
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import numpy

# -------------------------------Public Protocol-------------------------------
@typing.runtime_checkable
class DataBlockLike(typing.Protocol):
    @classmethod
    def load(cls, fpath: str) -> typing.Self: ...
    @property
    def meta(self) -> _BlockMeta:...
    @property
    def data(self) -> _DataLike:...

# private type
class _BlockMeta(typing.TypedDict):
    block_name: str
    valid_pixel_ratio: dict[str, float]
    has_label: bool
    label_nodata: int
    ignore_label: int
    label_num_classes: int
    label_to_ignore: list[int]
    label_class_name: typing.NotRequired[dict[str, str]]
    label_reclass_map: dict[str, list[int]]
    label_reclass_name: typing.NotRequired[dict[str, str]]
    label_count: dict[str, list[int]]
    label_entropy: dict[str, float]
    image_nodata: float
    dem_pad: int
    band_map: dict[str, int]
    spectral_indices_added: list[str]
    topo_metrics_added: list[str]
    block_image_stats: dict[str, dict[str, int | float]]

# private protocol
class _DataLike(typing.Protocol):
    @property
    def image_normalized(self) -> numpy.ndarray:...
    @property
    def label_masked(self) -> numpy.ndarray:...

# factory
T = typing.TypeVar('T', bound=DataBlockLike)
def load_block(cls: type[T], fpath: str) -> T:
    return cls().load(fpath)
