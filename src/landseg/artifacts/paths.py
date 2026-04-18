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

'''
Canonical filesystem paths for project artifacts.

Defines structured access to dataset artifacts produced by ingestion,
preprocessing, and transformation pipelines. Provides strongly-typed
path builders for foundation data, model dev/test blocks, and
transformed outputs.
'''

# standard imports
from __future__ import annotations
import dataclasses
import datetime
import os

# artifacts file structure from data ingestion and preparation piplines
# <exp_root>/artifacts/
# │
# ├── foundation/
# │   │
# │   ├── world_grids/
# │   │   └── grid_row_<srow>_<orow>_col_<scol>_<ocol>.json *
# │   │
# │   ├── domain_knowledge/
# │   │   ├── <domain_name>.json *
# │   │   └── <domain_name>_tiles_<gid>.npz
# │   │
# │   └── data_blocks/
# │       │
# │       ├── model_dev/
# │       │   ├── blocks/
# │       │   ├── windows/
# │       │   │   └── windows_<gid>.json
# │       │   ├── catalog.json
# │       │   └── schema.json
# │       │
# │       └── test_holdout/
# │           ├── blocks/
# │           ├── windows/
# │           │   └── windows_<gid>.json
# │           ├── catalog.json
# │           └── schema.json
# │
# └── transform/
#     │
#     ├── train_blocks/
#     ├── val_blocks/
#     ├── test_blocks/
#     │
#     ├── block_splits_source.json
#     ├── block_splits_summary.json
#     ├── block_splits_transformed.json
#     │
#     ├── label_stats.json
#     ├── image_stats.json
#     │
#     └── schema.json
#
#   *: a **_meta.json sidecar file will be generated as well
#
# -----------------------------------------------------------------------------
# results file structure from each training run
# <exp_root>/results/
# │
# ├── exp_0001/
# │   │
# │   ├── checkpoints/
# │   │
# │   ├── logs/
# │   │
# │   ├── plots/
# │   │
# │   ├── preivews/
# │   │
# |   └── config.json
# |
# ...

# artifacts/
@dataclasses.dataclass
class ArtifactPaths:
    '''Root entrypoint for all artifact path namespaces.'''
    root: str

    @property
    def foundation(self):
        return FoundationPaths(os.path.join(self.root, 'foundation'))

    @property
    def transform(self):
        return TransformPaths(os.path.join(self.root, 'transform'))

# artifacts/foundation/
@dataclasses.dataclass
class FoundationPaths:
    '''Paths for foundational datasets and knowledge artifacts.'''
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

# artifacts/foundation/world_grids/
@dataclasses.dataclass
class _WorldGrids:
    '''Paths for spatial grid tile artifacts.'''
    root: str

    def fpath(self, tile_specs: tuple[int, int, int, int]) -> str:
        '''Return canonical grid artifact file path.'''
        srow, scol, orow, ocol = tile_specs
        gid = f'grid_row_{srow}_{orow}_col_{scol}_{ocol}'
        return os.path.join(self.root, f'{gid}.json')

# artifacts/foundation/domain_maps/
@dataclasses.dataclass
class _DomainMaps:
    '''Paths for domain knowledge maps and tile mappings.'''
    root: str

    def domain_map_fpath(self, domain_name: str) -> str:
        no_ext, _ = os.path.splitext(domain_name)
        return os.path.join(self.root, f'{no_ext}.json')

    def mapped_tiles_fpath(self, domain_name: str, gid: str) -> str:
        no_ext, _ = os.path.splitext(domain_name)
        return os.path.join(self.root, f'{no_ext}_tiles_{gid}.npz')

# artifacts/foundation/data_blocks/
@dataclasses.dataclass
class _DataBlocks:
    '''Root container for model dev and test data blocks.'''
    root: str

    @property
    def dev(self):
        return _DataBlockPaths(os.path.join(self.root, 'model_dev'))

    @property
    def test(self):
        return _DataBlockPaths(os.path.join(self.root, 'test_holdout'))

@dataclasses.dataclass
class _DataBlockPaths:
    '''Paths for a single data block partition (dev/test).'''
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

# artifacts/transform
@dataclasses.dataclass
class TransformPaths:
    '''Paths for transformed datasets and split artifacts.'''
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

# results
@dataclasses.dataclass
class ResultsPaths:
    '''Root entry of a training run.'''

    results_root: str
    run_folder: str = ''

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.run_folder, 'checkpoints')

    @property
    def phase_status(self) -> str:
        return os.path.join(self.checkpoints, 'status.json')

    @property
    def logs(self) -> str:
        return os.path.join(self.run_folder, 'logs')

    @property
    def main_log_file(self) -> str:
        # e.g., 20001234_567
        t_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.logs, f'main_{t_stamp}.log')

    @property
    def plots(self) -> str:
        return os.path.join(self.run_folder, 'plots')

    @property
    def previews(self) -> str:
        return os.path.join(self.run_folder, 'previews')

    @property
    def config(self) -> str:
        return os.path.join(self.run_folder, 'config.json')

    def init(self):
        '''Initialize a results folder.'''

        # find the latest run number
        i = 1
        while True:
            self.run_folder = os.path.join(self.results_root, f'run_{i:04d}')
            try:
                os.makedirs(self.run_folder)
                break
            except FileExistsError:
                i += 1

        # create all subfolders if not already exist
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        os.makedirs(self.plots, exist_ok=True)
        os.makedirs(self.previews, exist_ok=True)

    def best_checkpoint(self, phase_name) -> str:
        return os.path.join(self.checkpoints, f'{phase_name}_best.pt')

    def last_checkpoint(self, phase_name) -> str:
        return os.path.join(self.checkpoints, f'{phase_name}_last.pt')
