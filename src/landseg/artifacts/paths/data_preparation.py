# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Canonical filesystem paths for data preparation artifacts.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass
class PreparationPaths(base.PipelineArtifactsPaths):
    '''Paths for prepared datasets and split artifacts.'''

    @property
    def report(self) -> str:
        '''Return the file path of the preparation execution report.'''
        return os.path.join(self.root, 'prep_report.json')

    @property
    def config(self) -> str:
        '''Return the file path of the persisted preparation configuration.'''
        return os.path.join(self.root, 'config.json')

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
    def splits_prepared_blocks(self) -> str:
        return os.path.join(self.root, 'block_splits_prepared.json')

    @property
    def schema(self) -> str:
        return os.path.join(self.root, 'schema.json')

    def _init_pipeline_folders(self):
        os.makedirs(self.train_blocks, exist_ok=True)
        os.makedirs(self.val_blocks, exist_ok=True)
        os.makedirs(self.test_blocks, exist_ok=True)
