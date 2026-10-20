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
Root entrypoint for all artifact path namespaces.
'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.paths as paths
import landseg.configs as configs


# ----- `ArtifactPaths` definition
@dataclasses.dataclass
class ArtifactPaths:
    '''Root entrypoint for all artifact path namespaces.'''
    root: pathlib.Path = pathlib.Path('./experiment')
    world_grid_root: pathlib.Path | None = None
    harmonization_root: pathlib.Path | None = None
    ingestion_root: pathlib.Path | None = None
    preparation_root: pathlib.Path | None = None
    session_root: pathlib.Path | None = None
    knowledge_root: pathlib.Path = pathlib.Path('knowledge')

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        if self.world_grid_root is not None:
            self.world_grid_root = pathlib.Path(self.world_grid_root)
        if self.harmonization_root is not None:
            self.harmonization_root = pathlib.Path(self.harmonization_root)
        if self.ingestion_root is not None:
            self.ingestion_root = pathlib.Path(self.ingestion_root)
        if self.preparation_root is not None:
            self.preparation_root = pathlib.Path(self.preparation_root)
        if self.session_root is not None:
            self.session_root = pathlib.Path(self.session_root)
        self.knowledge_root = pathlib.Path(self.knowledge_root)

    @classmethod
    def from_config(cls, config: 'configs.RootConfig') -> typing.Self:
        '''Construct `ArtifactPaths` from a `RootConfig` instance.'''
        return cls(
            root=pathlib.Path(config.execution.exp_root),
            world_grid_root=(
                pathlib.Path(config.data.world_grid.output_dpath)
                if config.data.world_grid.output_dpath
                else None
            ),
            harmonization_root=(
                pathlib.Path(config.data.harmonization.output_dpath)
                if config.data.harmonization.output_dpath
                else None
            ),
            ingestion_root=(
                pathlib.Path(config.data.ingestion.output_dpath)
                if config.data.ingestion.output_dpath
                else None
            ),
            preparation_root=(
                pathlib.Path(config.data.preparation.output_dpath)
                if config.data.preparation.output_dpath
                else None
            ),
            session_root=(
                pathlib.Path(config.session.output_dpath)
                if config.session.output_dpath
                else None
            ),
        )

    @property
    def world_grid(self) -> paths.WorldGridPaths:
        r = self.world_grid_root or (self.root / 'world_grids')
        return paths.WorldGridPaths(r)

    @property
    def data_harmonization(self) -> paths.HarmonizationPaths:
        r = self.harmonization_root or (self.root / 'harmonized_data')
        return paths.HarmonizationPaths(r)

    @property
    def data_ingestion(self) -> paths.IngestionPaths:
        r = self.ingestion_root or (self.root / 'ingested_data')
        return paths.IngestionPaths(r)

    @property
    def data_preparation(self) -> paths.PreparationPaths:
        r = self.preparation_root or (self.root / 'prepared_data')
        return paths.PreparationPaths(r)

    @property
    def session(self) -> paths.SessionPaths:
        r = self.session_root or (self.root / 'results')
        return paths.SessionPaths(root=r)

    @property
    def knowledge(self) -> paths.KnowledgePaths:
        return paths.KnowledgePaths(self.knowledge_root)
