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
Top-level namespace for `landseg.notebooks`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.

Public APIs:
    - `NotebookIOConfig`: Container for resolved notebook I/O paths.
    - `NotebookWidgets`: Ergonomic widget wrapper across runtimes.
    - `is_databricks`: Probe whether runtime is inside Databricks.
    - `is_notebook`: Probe whether code is running inside a notebook.
    - `resolve_exp_root`: Resolve experiment root path across contexts.
    - `resolve_notebook_io`: Resolve unified notebook I/O structure.
    - `resolve_optuna_storage`: Resolve environment-safe Optuna URI.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'NotebookIOConfig',
    'NotebookWidgets',
    # functions
    'is_databricks',
    'is_notebook',
    'resolve_exp_root',
    'resolve_notebook_io',
    'resolve_optuna_storage',
]


# for static check
if typing.TYPE_CHECKING:
    from .env import (
        is_databricks,
        is_notebook,
    )
    from .io import (
        NotebookIOConfig,
        resolve_exp_root,
        resolve_notebook_io,
        resolve_optuna_storage,
    )
    from .widgets import (
        NotebookWidgets,
    )


def __getattr__(name: str):
    if name in {'NotebookWidgets'}:
        obj = importlib.import_module('.widgets', __package__)
        return getattr(obj, name)

    if name in {
        'NotebookIOConfig',
        'resolve_exp_root',
        'resolve_notebook_io',
        'resolve_optuna_storage',
    }:
        obj = importlib.import_module('.io', __package__)
        return getattr(obj, name)

    if name in {'is_databricks', 'is_notebook'}:
        obj = importlib.import_module('.environment', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
