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
Canonical filesystem paths for data ingestion artifacts.
'''


# standard imports
import dataclasses
import os
# local imports
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass
class IngestionPaths(base.PipelineArtifactsPaths):
    '''Paths for ingested datasets and knowledge artifacts.'''

    @property
    def data_blocks(self):
        return _DataBlocks(os.path.join(self.root, 'data_blocks'))

    @property
    def domains(self):
        return _DomainMaps(os.path.join(self.root, 'domain_knowledge'))

    @property
    def report(self) -> str:
        return os.path.join(self.effective_run_folder, 'ingest_report.json')

    @property
    def config(self) -> str:
        return os.path.join(self.effective_run_folder, 'config.json')

    def _init_pipeline_folders(self):
        os.makedirs(self.effective_run_folder, exist_ok=True)
        os.makedirs(self.data_blocks.root, exist_ok=True)
        os.makedirs(self.data_blocks.blocks, exist_ok=True)
        os.makedirs(self.data_blocks.windows, exist_ok=True)
        os.makedirs(self.domains.root, exist_ok=True)


# ----- private dataclasses
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


@dataclasses.dataclass
class _DataBlocks:
    '''Container for canonical dataset data blocks.'''
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
