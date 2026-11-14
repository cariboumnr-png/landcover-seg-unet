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

Each stage namespace resolves from its explicit directory root. The
legacy shared `root` is optional and used only as a fallback when a
stage root is not configured, so standalone decoupled execution never
requires a common experiment parent.

Public APIs:
    - `ArtifactPaths`: Aggregation facade over stage path namespaces.
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
    root: pathlib.Path | None = None
    world_grid_root: pathlib.Path | None = None
    harmonization_root: pathlib.Path | None = None
    ingestion_root: pathlib.Path | None = None
    preparation_root: pathlib.Path | None = None
    session_root: pathlib.Path | None = None
    knowledge_root: pathlib.Path | None = None

    def __post_init__(self):
        if self.root is not None:
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
        if self.knowledge_root is not None:
            self.knowledge_root = pathlib.Path(self.knowledge_root)

    @classmethod
    def from_config(cls, config: 'configs.RootConfig') -> typing.Self:
        '''Construct `ArtifactPaths` from a `RootConfig` instance.'''
        exp_root = config.execution.exp_root
        return cls(
            root=pathlib.Path(exp_root) if exp_root else None,
            world_grid_root=_to_path(config.data.world_grid.output_dpath),
            harmonization_root=_to_path(
                config.data.harmonization.output_dpath
            ),
            ingestion_root=_to_path(config.data.ingestion.output_dpath),
            preparation_root=_to_path(
                config.data.preparation.output_dpath
            ),
            session_root=_to_path(config.session.output_dpath),
            knowledge_root=_to_path(config.execution.knowledge_dir),
        )

    @property
    def world_grid(self) -> paths.WorldGridPaths:
        r = _stage_root(
            self.world_grid_root,
            self.root,
            pathlib.Path('artifacts') / 'world_grids',
            'data.world_grid.output_dpath',
        )
        return paths.WorldGridPaths(r)

    @property
    def data_harmonization(self) -> paths.HarmonizationPaths:
        r = _stage_root(
            self.harmonization_root,
            self.root,
            pathlib.Path('artifacts') / 'harmonized_data',
            'data.harmonization.output_dpath',
        )
        return paths.HarmonizationPaths(r)

    @property
    def data_ingestion(self) -> paths.IngestionPaths:
        r = _stage_root(
            self.ingestion_root,
            self.root,
            pathlib.Path('artifacts') / 'ingested_data',
            'data.ingestion.output_dpath',
        )
        return paths.IngestionPaths(r)

    @property
    def data_preparation(self) -> paths.PreparationPaths:
        r = _stage_root(
            self.preparation_root,
            self.root,
            pathlib.Path('artifacts') / 'prepared_data',
            'data.preparation.output_dpath',
        )
        return paths.PreparationPaths(r)

    @property
    def session(self) -> paths.SessionPaths:
        r = _stage_root(
            self.session_root,
            self.root,
            pathlib.Path('results'),
            'session.output_dpath',
        )
        return paths.SessionPaths(root=r)

    @property
    def knowledge(self) -> paths.KnowledgePaths | None:
        if self.knowledge_root is None:
            return None
        return paths.KnowledgePaths(self.knowledge_root)


# ----- private helpers
def _to_path(value: str | pathlib.Path | None) -> pathlib.Path | None:
    '''Normalize an optional configured path value.'''
    if value is None or value == '':
        return None
    return pathlib.Path(value)


def _stage_root(
    explicit: pathlib.Path | None,
    shared_root: pathlib.Path | None,
    fallback: pathlib.Path,
    config_key: str,
) -> pathlib.Path:
    '''
    Resolve one stage root.

    Explicit stage roots always win. The legacy shared root is only a
    fallback when provided; otherwise the caller must configure the
    stage path explicitly.
    '''
    if explicit is not None:
        return explicit
    if shared_root is not None:
        return shared_root / fallback
    raise ValueError(
        f'No root configured for this artifact namespace. Set '
        f'"{config_key}" explicitly, or provide a shared "root"/'
        f'"execution.exp_root" to derive it.'
    )
