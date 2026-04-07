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
        return FoundationPaths(os.path.join(self.root, 'transform'))

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

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _WorldGrids:
    '''doc'''
    root: str

@dataclasses.dataclass
class _DomainMaps:
    '''doc'''
    root: str

@dataclasses.dataclass
class _DataBlocks:
    '''doc'''
    root: str

    @property
    def dev_root(self) -> str:
        return os.path.join(self.root, 'model_dev')

    @property
    def dev_blocks(self) -> str:
        return os.path.join(self.dev_root, 'blocks')

    @property
    def dev_windows(self) -> str:
        return os.path.join(self.dev_root, 'windows')

    @property
    def test_root(self) -> str:
        return os.path.join(self.root, 'test_holdout')

    @property
    def test_blocks(self) -> str:
        return os.path.join(self.test_root, 'blocks')

    @property
    def test_windows(self) -> str:
        return os.path.join(self.test_root, 'windows')
