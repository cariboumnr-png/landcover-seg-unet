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

'''
Protocols for Epoch-level engine components.
'''

# standard imports
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import torch

# ---------------------------------dataloaders---------------------------------
@typing.runtime_checkable
class DataLoadersLike(typing.Protocol):
    @property
    def train(self) -> 'torch.utils.data.DataLoader | None':...
    @property
    def val(self) -> 'torch.utils.data.DataLoader | None':...
    @property
    def test(self) -> 'torch.utils.data.DataLoader | None':...
    @property
    def meta(self) -> _DataLoadersMeta: ...

class _DataLoadersMeta(typing.Protocol):
    @property
    def batch_size(self) -> int: ...
    @property
    def patch_size(self) -> int: ...
    @property
    def preview_context(self) -> '_PreviewContext | None': ...

class _PreviewContext(typing.Protocol):
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]
