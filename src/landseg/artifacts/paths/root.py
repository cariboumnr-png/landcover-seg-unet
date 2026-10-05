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
import os
import typing
# local imports
import landseg.artifacts.paths as paths

if typing.TYPE_CHECKING:
    import landseg.configs as configs


# ----- `ArtifactPaths` definition
@dataclasses.dataclass
class ArtifactPaths:
    '''Root entrypoint for all artifact path namespaces.'''
    root: str = './experiment'
    world_grid_root: str | None = None
    harmonization_root: str | None = None
    ingestion_root: str | None = None
    preparation_root: str | None = None
    session_root: str | None = None
    knowledge_root: str = 'knowledge'

    @classmethod
    def from_config(cls, config: 'configs.RootConfig') -> typing.Self:
        '''Construct `ArtifactPaths` from a `RootConfig` instance.'''
        return cls(
            root=config.execution.exp_root,
            harmonization_root=config.data.harmonization.output_dpath,
            ingestion_root=config.data.ingestion.output_dpath,
            preparation_root=config.data.preparation.output_dpath,
            session_root=config.session.output_dpath,
        )

    @property
    def world_grid(self) -> paths.WorldGridPaths:
        r = self.world_grid_root or os.path.join(self.root, 'world_grids')
        return paths.WorldGridPaths(r)

    @property
    def data_harmonization(self) -> paths.HarmonizationPaths:
        r = self.harmonization_root or os.path.join(self.root, 'harmonized_data')
        return paths.HarmonizationPaths(r)

    @property
    def data_ingestion(self) -> paths.IngestionPaths:
        r = self.ingestion_root or os.path.join(self.root, 'ingested_data')
        return paths.IngestionPaths(r)

    @property
    def data_preparation(self) -> paths.PreparationPaths:
        r = self.preparation_root or os.path.join(self.root, 'prepared_data')
        return paths.PreparationPaths(r)

    @property
    def session(self) -> paths.SessionPaths:
        r = self.session_root or os.path.join(self.root, 'results')
        return paths.SessionPaths(root=r)

    @property
    def knowledge(self) -> paths.KnowledgePaths:
        return paths.KnowledgePaths(self.knowledge_root)
