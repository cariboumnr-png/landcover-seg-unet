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
Top-level namespace for `landseg.mlops`.

Provides standardized enterprise MLOps capabilities across CLI and notebooks:
Unity Catalog Model Registry integration, runtime environment detection,
lineage harvesting, and cloud volume storage resolution.

Public APIs:
    - `get_cluster_id`: Probe active compute cluster identifier.
    - `get_dbutils`: Locate active dbutils instance.
    - `get_git_commit`: Retrieve git commit SHA of active repository.
    - `get_runtime_version`: Probe Databricks runtime version.
    - `is_databricks`: Probe whether runtime is inside Databricks.
    - `load_study_trials`: Load persisted sweep trials dataframe or summary.
    - `register_unity_catalog_model`: Register checkpoint to Unity Catalog.
    - `resolve_exp_root`: Resolve experiment root path across contexts.
    - `resolve_optuna_storage`: Resolve environment-safe Optuna URI.
    - `sync_optuna_study`: Persist Optuna study trials to volume storage.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # functions
    'get_cluster_id',
    'get_dbutils',
    'get_git_commit',
    'get_runtime_version',
    'is_databricks',
    'load_study_trials',
    'register_unity_catalog_model',
    'resolve_exp_root',
    'resolve_optuna_storage',
    'sync_optuna_study',
]

# for static check
if typing.TYPE_CHECKING:
    from .env import (
        get_cluster_id,
        get_dbutils,
        get_git_commit,
        get_runtime_version,
        is_databricks,
    )
    from .registry import (
        register_unity_catalog_model,
    )
    from .storage import (
        load_study_trials,
        resolve_exp_root,
        resolve_optuna_storage,
        sync_optuna_study,
    )


def __getattr__(name: str):
    if name in {
        'get_cluster_id',
        'get_dbutils',
        'get_git_commit',
        'get_runtime_version',
        'is_databricks',
    }:
        obj = importlib.import_module('.env', __package__)
        return getattr(obj, name)

    if name in {'register_unity_catalog_model'}:
        obj = importlib.import_module('.registry', __package__)
        return getattr(obj, name)

    if name in {
        'load_study_trials',
        'resolve_exp_root',
        'resolve_optuna_storage',
        'sync_optuna_study',
    }:
        obj = importlib.import_module('.storage', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
