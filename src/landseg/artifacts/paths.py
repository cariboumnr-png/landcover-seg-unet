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

# pylint: disable=missing-function-docstring

'''Project level artifact canonical file paths.'''

# standard imports
from __future__ import annotations
import dataclasses
import os

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ArtifactPaths:
    '''Doc'''
    root: str

    @property
    def foundation(self):
        return FoundationPaths(os.path.join(self.root, 'foundation'))

    @property
    def transform(self):
        return TransformPaths(os.path.join(self.root, 'transform'))

@dataclasses.dataclass
class FoundationPaths:
    '''doc'''
    root: str

    @property
    def grids(self):
        return _WorldGrids(os.path.join(self.root, 'world_grids'))

    @property
    def domains(self):
        return _DomainMaps(os.path.join(self.root, 'domain_knowledge'))

    @property
    def data_blocks(self):
        return _DataBlocks(os.path.join(self.root, 'data_blocks'))

@dataclasses.dataclass
class _WorldGrids:
    '''doc'''
    root: str

    def fpath(self, tile_specs: tuple[int, int, int, int]) -> str:
        '''Return canonical grid artifact file path.'''
        srow, scol, orow, ocol = tile_specs
        gid = f'grid_row_{srow}_{orow}_col_{scol}_{ocol}'
        return os.path.join(self.root, f'{gid}.json')

@dataclasses.dataclass
class _DomainMaps:
    '''doc'''
    root: str

    def domain_map_fpath(self, domain_name: str) -> str:
        no_ext, _ = os.path.splitext(domain_name)
        return os.path.join(self.root, f'{no_ext}.json')

    def mapped_tiles_fpath(self, domain_name: str, gid: str) -> str:
        no_ext, _ = os.path.splitext(domain_name)
        return os.path.join(self.root, f'{no_ext}_tiles_{gid}.npz')

@dataclasses.dataclass
class _DataBlocks:
    '''doc'''
    root: str

    @property
    def dev(self):
        return _DataBlockPaths(os.path.join(self.root, 'model_dev'))

    @property
    def test(self):
        return _DataBlockPaths(os.path.join(self.root, 'test_holdout'))

@dataclasses.dataclass
class _DataBlockPaths:
    '''doc'''
    root: str

    @property
    def blocks(self) -> str:
        return os.path.join(self.root, 'blocks')

    @property
    def windows(self) -> str:
        return os.path.join(self.root, 'windows')

    @property
    def catalog(self) -> str:
        return os.path.join(self.root, 'catalog.json')

    @property
    def schema(self) -> str:
        return os.path.join(self.root, 'schema.json')

    def mapped_window(self, gid: str) -> str:
        return os.path.join(self.windows, f'windows_{gid}.json')


@dataclasses.dataclass
class TransformPaths:
    '''doc'''
    root: str

    @property
    def train_blocks(self) -> str:
        return os.path.join(self.root, 'train_blocks')

    @property
    def val_blocks(self) -> str:
        return os.path.join(self.root, 'val_blocks')

    @property
    def test_blocks(self) -> str:
        return os.path.join(self.root, 'test_blocks')

    @property
    def splits_source_blocks(self) -> str:
        return os.path.join(self.root, 'block_splits_source.json')

    @property
    def splits_summary(self) -> str:
        return os.path.join(self.root, 'block_splits_summary.json')

    @property
    def label_stats(self) -> str:
        return os.path.join(self.root, 'label_stats.json')

    @property
    def image_stats(self) -> str:
        return os.path.join(self.root, 'image_stats.json')

    @property
    def splits_transformed_blocks(self) -> str:
        return os.path.join(self.root, 'block_splits_transformed.json')

    @property
    def schema(self) -> str:
        return os.path.join(self.root, 'schema.json')
