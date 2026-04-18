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
Data specifications interface.

This module defines the `DataSpecs` dataclass, a canonical, structured
representation of dataset configuration and statistics used across the
project.

`DataSpecs` objects are:
    - **Produced** by data preparation pipelines in the `geopipe` module,
      where raw data is processed into model-ready artifacts.
    - **Consumed** during model construction, providing required shape,
      label topology, and domain information.
    - **Used at runtime** by trainers and evaluators to ensure consistent
      interpretation of inputs, outputs, and dataset splits.

As part of the `./core/` package, this interface serves as a stable
contract between data pipelines and downstream training/inference
components.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

# CONSTANTS
SPEC_BAND_NAMES = {
    'red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'ndvi', 'ndmi', 'nbr'
}
TOPO_BAND_NAMES = {
    'dem', 'slope', 'cos_aspect', 'sin_aspect', 'tpi'
}

# alias
field = dataclasses.field

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataSpecs:
    '''Container for dataset specs used by trainers and models'''
    name: str
    mode: typing.Literal['default', 'single', 'val_only', 'test_only']
    meta: Meta         # general dataset metadata
    heads: Heads       # head-wise label statistics and topology
    splits: Splits     # train/val/test block file mappings
    domains: Domains   # discrete/continuous domain metadata for conditioning

    def __str__(self) -> str:
        return '\n'.join([
            'Data Specifications:\n----------------------------------------',
            f'DataSpecs name: {self.name}',
            f'DataSpecs mode: {self.mode}',
            str(self.meta),
            str(self.heads),
            str(self.splits),
            str(self.domains)
        ])

@dataclasses.dataclass
class Meta:
    '''General dataset metadata.'''

    @dataclasses.dataclass
    class Image:
        '''Image related.'''
        num_channels: int
        height_width: int
        array_key: str
        band_map: dict[str, int]
        spec_channels: list[int] = field(default_factory=list)
        topo_channels: list[int] = field(default_factory=list)

        def __post_init__(self):
            # simple grouping by name for now
            for k, v in self.band_map.items():
                if k.lower() in SPEC_BAND_NAMES:
                    self.spec_channels.append(v)
                elif k.lower in TOPO_BAND_NAMES:
                    self.topo_channels.append(v)

    @dataclasses.dataclass
    class Label:
        '''Label related.'''
        ignore_index: int
        array_key: str

    # general - currently unclassified
    blk_bytes: int
    test_blks_grid: tuple[int, int]
    # image specs
    image_specs: Image
    # lable specs
    label_specs: Label

    def __str__(self) -> str:
        s = self.image_specs.height_width
        return '\n'.join([
            '[General Meta]',
            f'Number of image channels: {self.image_specs.num_channels}',
            f'Data block size (H==W): {s, s}',
            f'Ignore index: {self.label_specs.ignore_index}',
            f'Per-block byte size: {self.blk_bytes:,}',
            f'Test blocks grid: {self.test_blks_grid}'
        ])

@dataclasses.dataclass
class Heads:
    '''Head-wise label statistics and topology.'''
    class_counts: dict[str, list[int]]
    logits_adjust: dict[str, list[float]]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]

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
    test: dict[str, str]

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
