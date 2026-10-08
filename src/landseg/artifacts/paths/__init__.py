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
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'ArtifactPaths',
    'HarmonizationPaths',
    'IngestionPaths',
    'KnowledgePaths',
    'PipelineArtifactsPaths',
    'PreparationPaths',
    'SessionPaths',
    'WorldGridPaths',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import (
        PipelineArtifactsPaths,
    )
    from .data_harmonization import (
        HarmonizationPaths,
    )
    from .data_ingestion import (
        IngestionPaths,
    )
    from .data_preparation import (
        PreparationPaths,
    )
    from .knowledge import (
        KnowledgePaths,
    )
    from .root import (
        ArtifactPaths,
    )
    from .session import (
        SessionPaths,
    )
    from .world_grid import (
        WorldGridPaths
    )


def __getattr__(name: str):
    if name in {'HarmonizationPaths'}:
        obj = importlib.import_module('.data_harmonization', __package__)
        return getattr(obj, name)

    if name in {'IngestionPaths'}:
        obj = importlib.import_module('.data_ingestion', __package__)
        return getattr(obj, name)

    if name in {'PreparationPaths'}:
        obj = importlib.import_module('.data_preparation', __package__)
        return getattr(obj, name)

    if name in {'KnowledgePaths'}:
        obj = importlib.import_module('.knowledge', __package__)
        return getattr(obj, name)

    if name in {'ArtifactPaths'}:
        obj = importlib.import_module('.root', __package__)
        return getattr(obj, name)

    if name in {'PipelineArtifactsPaths'}:
        obj = importlib.import_module('.base', __package__)
        return getattr(obj, name)

    if name in {'SessionPaths'}:
        obj = importlib.import_module('.session', __package__)
        return getattr(obj, name)

    if name in {'WorldGridPaths'}:
        obj = importlib.import_module('.world_grid', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
