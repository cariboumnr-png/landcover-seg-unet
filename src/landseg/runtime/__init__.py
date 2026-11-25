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
    - `CORE_N`: Default worker count resolved from environment or CPU.
    - `Logger`: Manages file and console logging handlers.
    - `MissingRequiredInputError`: Error for missing notebook inputs.
    - `NotebookWidgets`: Ergonomic widget wrapper across runtimes.
    - `ParallelExecutor`: Configurable process and thread task executor.
    - `get_cluster_id`: Probe active compute cluster identifier.
    - `get_dbutils`: Locate active dbutils instance.
    - `get_default_worker_count`: Resolve default worker parallelism.
    - `get_device_name`: Query human-readable name of execution device.
    - `get_git_commit`: Retrieve git commit SHA of active repository.
    - `get_runtime_version`: Probe Databricks runtime version.
    - `is_cpu_bf16_supported`: Probe whether CPU supports hardware BF16.
    - `is_databricks`: Probe whether runtime is inside Databricks.
    - `is_notebook`: Probe whether code executes in notebook shell.
    - `require_widget`: Enforce and validate required notebook widgets.
    - `resolve_amp_dtype`: Resolve safe autocast dtype for device.
    - `resolve_device`: Resolve hardware execution target string.
    - `safe_training_session`: Interruption-safe training context manager.
    - `validator`: Parameter and directory validation routines.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'CORE_N',
    'Logger',
    'MissingRequiredInputError',
    'NotebookWidgets',
    'ParallelExecutor',
    'get_cluster_id',
    'get_dbutils',
    'get_default_worker_count',
    'get_device_name',
    'get_git_commit',
    'get_runtime_version',
    'is_cpu_bf16_supported',
    'is_databricks',
    'is_notebook',
    'require_widget',
    'resolve_amp_dtype',
    'resolve_device',
    'safe_training_session',
    'validator',
]


# for static check
if typing.TYPE_CHECKING:
    from . import validator
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
        safe_training_session,
    )
    from .logger import Logger
    from .platform import (
        get_cluster_id,
        get_dbutils,
        get_git_commit,
        get_runtime_version,
        is_databricks,
        is_notebook,
    )


__getattr__, __dir__ = _lazy.attach(
    __name__,
    {
        '.concurrency': [
            'CORE_N',
            'ParallelExecutor',
            'get_default_worker_count',
        ],
        '.hardware': [
            'get_device_name',
            'is_cpu_bf16_supported',
            'resolve_amp_dtype',
            'resolve_device',
        ],
        '.interactive': [
            'MissingRequiredInputError',
            'NotebookWidgets',
            'require_widget',
            'safe_training_session',
        ],
        '.logger': ['Logger'],
        '.platform': [
            'get_cluster_id',
            'get_dbutils',
            'get_git_commit',
            'get_runtime_version',
            'is_databricks',
            'is_notebook',
        ],
    },
    submodules=['validator'],
)
