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
import pathlib
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass
class IngestionPaths(base.PipelineArtifactsPaths):
    '''Paths for ingested datasets and knowledge artifacts.'''

    @property
    def runs_manifest(self) -> pathlib.Path:
        return pathlib.Path(self.root) / 'ingestion_runs.json'

    @property
    def data_blocks(self) -> '_DataBlocks':
        return _DataBlocks(pathlib.Path(self.root) / 'data_blocks')

    @property
    def domains(self) -> '_DomainMaps':
        return _DomainMaps(pathlib.Path(self.root) / 'domain_knowledge')

    @property
    def windows(self) -> pathlib.Path:
        return pathlib.Path(self.root) / 'block_windows.json'

    @property
    def collisions(self) -> pathlib.Path:
        return self.effective_run_folder / 'collisions.json'

    def _init_pipeline_folders(self):
        super()._init_pipeline_folders()
        pathlib.Path(self.data_blocks.root).mkdir(parents=True, exist_ok=True)
        self.data_blocks.blocks.mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.domains.root).mkdir(parents=True, exist_ok=True)
        if not self.runs_manifest.exists():
            controller.Controller[dict](self.runs_manifest).persist({})


# ----- private dataclasses
@dataclasses.dataclass
class _DomainMaps:
    '''Paths for domain knowledge maps and tile mappings.'''
    root: pathlib.Path | str

    def __post_init__(self):
        self.root = pathlib.Path(self.root)

    def domain_map_fpath(self, domain_name: str) -> pathlib.Path:
        stem = pathlib.Path(domain_name).stem
        return self.root / f'{stem}.json'

    def mapped_tiles_fpath(self, domain_name: str, gid: str) -> pathlib.Path:
        stem = pathlib.Path(domain_name).stem
        return self.root / f'{stem}_tiles_{gid}.npz'


@dataclasses.dataclass
class _DataBlocks:
    '''Container for canonical dataset data blocks.'''
    root: pathlib.Path | str

    def __post_init__(self):
        self.root = pathlib.Path(self.root)

    @property
    def blocks(self) -> pathlib.Path:
        return self.root / 'blocks'

    @property
    def catalog(self) -> pathlib.Path:
        return self.root / 'catalog.json'

    @property
    def schema(self) -> pathlib.Path:
        return self.root / 'schema.json'
