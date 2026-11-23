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
Top-level namespace for `landseg.runtime`.

Provides environment detection, hardware capability resolution,
concurrency orchestration, logging, and notebook abstractions.

Public APIs:
    - `Logger`: Manages file and console logging handlers.
    - `MissingRequiredInputError`: Error raised on missing notebook parameters.
    - `NotebookWidgets`: Ergonomic widget wrapper across runtimes.
    - `ParallelExecutor`: Configurable process and thread task executor.
    - `get_default_worker_count`: Resolve default worker parallelism.
    - `get_device_name`: Query human-readable name of execution device.
    - `get_cluster_id`: Probe active compute cluster identifier.
    - `get_dbutils`: Locate active dbutils instance.
    - `get_git_commit`: Retrieve git commit SHA of active repository.
    - `get_runtime_version`: Probe Databricks runtime version.
    - `is_notebook`: Probe whether code is running inside a notebook.
    - `is_cpu_bf16_supported`: Probe whether CPU supports hardware BF16.
    - `is_databricks`: Probe whether runtime is inside Databricks.
    - `is_notebook`: Probe whether code executes in notebook shell.
    - `resolve_device`: Resolve hardware execution target string.
    - `require_widget`: Enforce and validate required notebook widgets.
    - `safe_training_session`: Interruption-safe training context manager.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # constants
    'CORE_N',
    # classes
    'Logger',
    'MissingRequiredInputError',
    'NotebookWidgets',
    'ParallelExecutor',
    # funtions
    'get_default_worker_count',
    'get_device_name',
    'get_cluster_id',
    'get_dbutils',
    'get_git_commit',
    'get_runtime_version',
    'is_cpu_bf16_supported',
    'is_databricks',
    'is_notebook',
    'require_widget',
    'resolve_device',
    'safe_training_session',
]


# for static check
if typing.TYPE_CHECKING:
    from .concurrency import (
        CORE_N,
        ParallelExecutor,
        get_default_worker_count,
    )
    from .hardware import (
        get_device_name,
        is_cpu_bf16_supported,
        resolve_amp_dtype,
        resolve_device,
    )
    from .interactive import (
        MissingRequiredInputError,
        NotebookWidgets,
        require_widget,
    )
    from .logger import (
        Logger,
    )
    from .platform import (
        get_cluster_id,
        get_dbutils,
        get_git_commit,
        get_runtime_version,
        is_databricks,
        is_notebook,
    )
    from .session import (
        safe_training_session,
    )


def __getattr__(name: str) -> typing.Any:
    if name in {
        'CORE_N',
        'ParallelExecutor',
        'get_default_worker_count'
    }:
        obj = importlib.import_module('.concurrency', __package__)
        return getattr(obj, name)

    if name in {
        'get_device_name',
        'is_cpu_bf16_supported',
        'resolve_amp_dtype',
        'resolve_device',
    }:
        obj = importlib.import_module('.hardware', __package__)
        return getattr(obj, name)

    if name in {
        'MissingRequiredInputError',
        'NotebookWidgets',
        'require_widget',
    }:
        obj = importlib.import_module('.interactive', __package__)
        return getattr(obj, name)

    if name in {'Logger'}:
        obj = importlib.import_module('.logger', __package__)
        return getattr(obj, name)

    if name in {
        'get_cluster_id',
        'get_dbutils',
        'get_git_commit',
        'get_runtime_version',
        'is_databricks',
        'is_notebook',
    }:
        obj = importlib.import_module('.platform', __package__)
        return getattr(obj, name)

    if name in {'safe_training_session'}:
        obj = importlib.import_module('.session', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
