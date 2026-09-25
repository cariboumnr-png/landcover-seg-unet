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
Top-level namespace for `landseg.artifacts`.

Exposes selected public symbols via lazy resolution to keep import
order simple and circular-free.

Public APIs:
    - `ArtifactError`: Base artifact operational error.
    - `ArtifactPaths`: Generic artifact paths resolver.
    - `CheckpointMeta`: Metadata TypedDict for model checkpoints.
    - `Controller`: Base artifact controller.
    - `HarmonizationPaths`: Harmonization artifact directory paths.
    - `IngestionPaths`: Ingestion artifact directory paths.
    - `KnowledgePaths`: Knowledge artifact directory paths.
    - `LifecyclePolicy`: Artifact lifecycle retention policy.
    - `PayloadController`: Base JSON/dict payload controller.
    - `PayloadDict`: TypedDict mapping keys to JSON-serializable payloads.
    - `PreparationPaths`: Preparation artifact directory paths.
    - `SessionPaths`: Session artifact directory paths.
    - `check_run_collision`: Search runs manifest for matching run.
    - `compute_fingerprint`: Deterministic SHA-256 hash of payload.
    - `load_checkpoint`: Restore model and optimizer checkpoint.
    - `save_checkpoint`: Persist model and optimizer checkpoint.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'ArtifactError',
    'ArtifactPaths',
    'Controller',
    'HarmonizationPaths',
    'IngestionPaths',
    'KnowledgePaths',
    'LifecyclePolicy',
    'PayloadController',
    'PreparationPaths',
    'SessionPaths',
    # functions
    'check_run_collision',
    'compute_fingerprint',
    'load_checkpoint',
    'save_checkpoint',
    # typing
    'CheckpointMeta',
    'PayloadDict',
]


# for static check
if typing.TYPE_CHECKING:
    from .checkpoint import (
        CheckpointMeta,
        load_checkpoint,
        save_checkpoint,
    )
    from .controller import (
        ArtifactError,
        Controller,
    )
    from .ledger import (
        check_run_collision,
        compute_fingerprint,
    )
    from .paths import (
        ArtifactPaths,
        HarmonizationPaths,
        IngestionPaths,
        KnowledgePaths,
        PreparationPaths,
        SessionPaths,
    )
    from .payload_io import (
        PayloadController,
        PayloadDict,
    )
    from .policy import (
        LifecyclePolicy,
    )


def __getattr__(name: str):
    if name in {
        'CheckpointMeta',
        'load_checkpoint',
        'save_checkpoint',
    }:
        obj = importlib.import_module('.checkpoint', __package__)
        return getattr(obj, name)

    if name in {
        'ArtifactError',
        'Controller',
    }:
        obj = importlib.import_module('.controller', __package__)
        return getattr(obj, name)

    if name in {
        'check_run_collision',
        'compute_fingerprint',
    }:
        obj = importlib.import_module('.ledger', __package__)
        return getattr(obj, name)

    if name in {
        'ArtifactPaths',
        'HarmonizationPaths',
        'IngestionPaths',
        'KnowledgePaths',
        'PreparationPaths',
        'SessionPaths',
    }:
        obj = importlib.import_module('.paths', __package__)
        return getattr(obj, name)

    if name in {
        'PayloadController',
        'PayloadDict',
    }:
        obj = importlib.import_module('.payload_io', __package__)
        return getattr(obj, name)

    if name in {'LifecyclePolicy'}:
        obj = importlib.import_module('.policy', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
