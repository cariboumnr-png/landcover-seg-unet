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

'''Typed dictionaries for JSON based artifacts'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------------Public Type---------------------------------
class BlockSplitPaths(typing.TypedDict):
    '''Original data block files from the catalog.'''
    train: list[str]
    val: list[str]
    test: list[str]

class ImageBandStats(typing.TypedDict):
    '''Per band image stats.'''
    total_count: int
    current_mean: float
    accum_m2: float
    std: float

class LabelStats(typing.TypedDict):
    '''Typed label stats.'''
    original_counts: list[int]
    original_proportions: list[float]
    current_counts: list[int]
    current_proportions: list[float]

class SchemaFull(typing.TypedDict):
    '''Dataset-wide schema emitted by `build_schema_full(...)`.'''
    schema_version: str
    creation_time: str
    artifacts: dict[str, str]
    checksums: dict[str, str]
    train_blocks: list[str]
    val_blocks: list[str]
    test_blocks: list[str]
    label_stats: LabelStats
    image_stats: dict[str, ImageBandStats]

class SchemaOneBlock(typing.TypedDict):
    '''Minimal schema emitted by `build_schema_one_block(...)`.'''
    dataset_name: str
    image_channel: int
    image_h_w: int
    ignore_index: int
    img_arr_key: str
    lbl_arr_key: str
    class_counts: dict[str, list[int]]
    logit_adjust: dict[str, list[float]]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]
    train_split: dict[str, str]
    val_split: dict[str, str]
