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

# pylint: disable=missing-function-docstring, too-few-public-methods
'''Dataset specification protocol used by models and trainers.'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------------Public Type---------------------------------
class DataSpecsLike(typing.Protocol):
    '''Protocol describing the full dataset specification interface.'''
    @property
    def meta(self) -> _Meta:...
    @property
    def heads(self) -> _Head:...
    @property
    def splits(self) -> _Splits: ...
    @property
    def domains(self) -> _Domains: ...

# --------------------------------private  type--------------------------------
class _Meta(typing.Protocol):
    '''Metadata section of the dataset specification.'''
    dataset_name: str
    img_ch_num: int
    ignore_index: int
    block_size: int
    fit_perblk_bytes: int
    test_blks_grid: tuple[int, int]
    single_block_mode: bool

class _Head(typing.Protocol):
    '''Head configuration section of the dataset specification.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, typing.Any]]

class _Splits(typing.Protocol):
    '''Split definitions for train/val/test loaders.'''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str] | None

class _Domains(typing.Protocol):
    '''Domain information for each dataset split.'''
    train: Dom
    val: Dom
    test: Dom
    ids_max: int
    vec_dim: int

    class Dom(typing.TypedDict):
        '''TypedDict describing a single domain entry.'''
        ids_domain: dict[str, int] | None
        vec_domain: dict[str, list[float]] | None
