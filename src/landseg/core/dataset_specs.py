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
DataSpecs interface
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataSpecs:
    '''Container for dataset specs used by trainers and models'''
    meta: Meta         # general dataset metadata
    heads: Heads       # head-wise label statistics and topology
    splits: Splits     # train/val/test block file mappings
    domains: Domains   # discrete/continuous domain metadata for conditioning

    def __str__(self) -> str:
        return '\n'.join([
            'Dataset summary:\n----------------------------------------',
            str(self.meta),
            str(self.heads),
            str(self.splits),
            str(self.domains)
        ])

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class Meta:
    '''General dataset metadata.'''
    dataset_name: str
    img_ch_num: int
    ignore_index: int
    block_size: int
    fit_perblk_bytes: int
    test_blks_grid: tuple[int, int]
    single_block_mode: bool

    def __str__(self) -> str:
        return '\n'.join([
            '[General Meta]',
            f'Dataset name: {self.dataset_name}',
            f'Number of image channels: {self.img_ch_num}',
            f'Ignore index: {self.ignore_index}'
        ])

@dataclasses.dataclass
class Heads:
    '''Head-wise label statistics and topology.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    topology: dict[str, dict[str, typing.Any]] # TODO need better typing

    def __str__(self) -> str:
        def _ln(lst):
            return [round(x, 2) for x in lst]
        cc = self.class_counts
        la = self.logits_adjust
        t1 = '\n - '.join([f'{k}:\t{v}' for k, v in cc.items()])
        t2 = '\n - '.join([f'{k}:\t{_ln(v)}' for k, v in la.items()])
        return '\n'.join([
            '[Heads Specs]',
            f'Class distribution of each head: \n - {t1}',
            f'Head-wise logits adjustment: \n - {t2}',
        ])

@dataclasses.dataclass
class Splits:
    '''Train/val/test block file mappings.'''
    train: dict[str, str]
    val: dict[str, str]
    test: dict[str, str] | None

    def __str__(self) -> str:
        return '\n'.join([
            '[Dataset Split]',
            f'Number of train blocks: {len(self.train)}',
            f'Number of val blocks: {len(self.val)}',
            f'Number of test blocks: {len(self.test or {})}',
        ])

@dataclasses.dataclass
class Domains:
    '''Domain metadata for conditioning.'''
    train: Dom
    val: Dom
    test: Dom
    ids_max: int
    vec_dim: int

    class Dom(typing.TypedDict):
        '''Typed domain dictionaries.'''
        ids_domain: dict[str, int] | None
        vec_domain: dict[str, list[float]] | None

    def __str__(self) -> str:
        return '\n'.join([
            '[Domain Knowledge]',
            f'Discrete domain IDs count: {self.ids_max + 1}', # 0- to 1-based
            f'Continuous domain PCA number of axes: {self.vec_dim}'
        ])
