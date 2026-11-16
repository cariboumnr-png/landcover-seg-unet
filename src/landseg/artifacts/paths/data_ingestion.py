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
import typing
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass
class IngestionPaths(base.PipelineArtifactsPaths):
    '''Paths for ingested datasets and knowledge artifacts.'''

    @property
    def runs_manifest(self) -> pathlib.Path:
        return self.root / 'ingestion_runs.json'

    @property
    def data_blocks(self) -> '_DataBlocks':
        return _DataBlocks(self.root / 'data_blocks')

    @property
    def domains(self) -> '_DomainMaps':
        return _DomainMaps(self.root / 'domain_knowledge')

    @property
    def windows(self) -> pathlib.Path:
        return self.root / 'block_windows.json'

    @property
    def collisions(self) -> pathlib.Path:
        return self.effective_run_folder / 'collisions.json'

    def _init_pipeline_folders(self):
        super()._init_pipeline_folders()
        self.data_blocks.root.mkdir(parents=True, exist_ok=True)
        self.data_blocks.blocks.mkdir(parents=True, exist_ok=True)
        self.domains.root.mkdir(parents=True, exist_ok=True)
        if not self.runs_manifest.exists():
            controller.Controller[dict](self.runs_manifest).persist({})

    def validate(
        self,
        require_domains: bool = True,
        require_blocks: bool = False,
        require_manifest: bool = False,
    ) -> typing.Self:
        '''
        Validate that ingestion artifact directories and schema exist.

        Args:
            require_domains:
                If True, validates that domain knowledge directory exists.
            require_blocks:
                If True, validates that data blocks pool directory exists.
            require_manifest:
                If True, validates that ingestion runs manifest exists.

        Returns:
            typing.Self: The validated instance for method chaining.

        Raises:
            ArtifactError:
                If expected directory structure or files are missing.
        '''
        super().validate()
        self.data_blocks.validate(require_blocks=require_blocks)
        if require_domains and not self.domains.root.is_dir():
            raise controller.ArtifactError(
                f'Domain knowledge directory does not exist '
                f'or is not a directory: {self.domains.root}'
            )
        if require_manifest and not self.runs_manifest.is_file():
            raise controller.ArtifactError(
                f'Ingestion runs manifest does not exist '
                f'or is not a file: {self.runs_manifest}'
            )
        return self


# ----- private dataclasses
@dataclasses.dataclass
class _DomainMaps:
    '''Paths for domain knowledge maps and tile mappings.'''
    root: pathlib.Path

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
    root: pathlib.Path

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

    def validate(self, require_blocks: bool = False) -> typing.Self:
        '''
        Validate that data blocks directory and canonical schema exist.

        Args:
            require_blocks:
                If True, also validates that the blocks directory exists.

        Returns:
            typing.Self: The validated instance for method chaining.

        Raises:
            ArtifactError:
                If expected directory structure or files are missing.
        '''
        if not self.root.is_dir():
            raise controller.ArtifactError(
                f'Data blocks directory does not exist or is not a directory: {self.root}'
            )
        if not self.schema.is_file():
            raise controller.ArtifactError(
                f'Data blocks schema file does not exist or is not a file: {self.schema}'
            )
        if require_blocks and not self.blocks.is_dir():
            raise controller.ArtifactError(
                f'Data blocks pool directory does not exist or is not a directory: {self.blocks}'
            )
        return self
