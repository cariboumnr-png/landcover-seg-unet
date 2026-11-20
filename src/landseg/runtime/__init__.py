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
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # concurrency
    'CORE_N',
    'ParallelExecutor',
    'get_default_worker_count',
    # hardware
    'get_device_name',
    'is_cpu_bf16_supported',
    'resolve_device',
    # interactive
    'MissingRequiredInputError',
    'NotebookWidgets',
    'require_widget',
    # logger
    'Logger',
    # platform
    'get_cluster_id',
    'get_dbutils',
    'get_git_commit',
    'get_runtime_version',
    'is_databricks',
    'is_notebook',
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


def __getattr__(name: str) -> typing.Any:
    if name in {'CORE_N', 'ParallelExecutor', 'get_default_worker_count'}:
        obj = importlib.import_module('.concurrency', __package__)
        return getattr(obj, name)

    if name in {
        'get_device_name',
        'is_cpu_bf16_supported',
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

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
