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

'''
Top-level namespace for `landseg.artifacts.paths`.

Exposes artifact and session path dataclasses for all pipelines.

Public APIs:
    - `ArtifactPaths`: Aggregation facade over stage path namespaces.
    - `HarmonizationPaths`: Paths for harmonization ETL artifacts.
    - `IngestionPaths`: Paths for data ingestion artifacts.
    - `KnowledgePaths`: Paths for knowledge base embeddings/profiles.
    - `PipelineArtifactsPaths`: Base paths dataclass for pipeline runs.
    - `PreparationPaths`: Paths for prepared datasets and split runs.
    - `RunScopedArtifactPaths`: Base handle for `run_XXXX` stages.
    - `SessionPaths`: Paths for training and evaluation session runs.
    - `WorldGridPaths`: Paths for canonical world grid artifacts.
    - `resolve_session_run`: Resolve experiment, results, or run path.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'ArtifactPaths',
    'HarmonizationPaths',
    'IngestionPaths',
    'KnowledgePaths',
    'PipelineArtifactsPaths',
    'PreparationPaths',
    'RunScopedArtifactPaths',
    'SessionPaths',
    'WorldGridPaths',
    'resolve_session_run',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import (
        PipelineArtifactsPaths,
        RunScopedArtifactPaths,
    )
    from .data_harmonization import HarmonizationPaths
    from .data_ingestion import IngestionPaths
    from .data_preparation import PreparationPaths
    from .knowledge import KnowledgePaths
    from .root import ArtifactPaths
    from .session import (
        SessionPaths,
        resolve_session_run,
    )
    from .world_grid import WorldGridPaths


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.base': [
        'PipelineArtifactsPaths',
        'RunScopedArtifactPaths',
    ],
    '.data_harmonization': ['HarmonizationPaths'],
    '.data_ingestion': ['IngestionPaths'],
    '.data_preparation': ['PreparationPaths'],
    '.knowledge': ['KnowledgePaths'],
    '.root': ['ArtifactPaths'],
    '.session': [
        'SessionPaths',
        'resolve_session_run',
    ],
    '.world_grid': ['WorldGridPaths'],
})
