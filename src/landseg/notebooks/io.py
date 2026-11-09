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
Notebook I/O path, directory layout, and storage backend utilities.

Provides precedence-based resolution of experiment root workspaces,
canonical artifact/input directory layouts, and safe Optuna storage
connection resolution across cloud and local runtimes.

Public APIs:
    - `NotebookIOConfig`: Container for resolved notebook I/O paths.
    - `resolve_exp_root`: Resolve experiment root path across contexts.
    - `resolve_notebook_io`: Resolve unified notebook I/O structure.
    - `resolve_optuna_storage`: Resolve environment-safe Optuna URI.
'''

# standard imports
import dataclasses
import os
import pathlib
import typing
# local imports
import landseg.notebooks.env as env
import landseg.notebooks.widgets as widgets_mod


# ----- public dataclasses
@dataclasses.dataclass
class NotebookIOConfig:
    '''Container for resolved notebook input and output directory paths.'''
    exp_root: pathlib.Path
    artifacts_dir: pathlib.Path
    results_dir: pathlib.Path
    input_dir: pathlib.Path


# ----- public functions
def resolve_exp_root(
    exp_root: str | pathlib.Path | None = None,
    default: str | pathlib.Path = './experiment',
    *,
    env_var: str = 'LANDSEG_EXP_ROOT',
    create_if_missing: bool = False,
    dbutils: typing.Any = None,
) -> pathlib.Path:
    '''
    Resolve experiment root path across local, CI, and Databricks.

    Evaluation precedence:
    1. Explicit `exp_root` argument if provided.
    2. Databricks notebook widget if available.
    3. Dedicated environment variable (default: `LANDSEG_EXP_ROOT`).
    4. Fallback default (`./experiment`).

    Args:
        exp_root:
            Optional explicit directory path.
        default:
            Fallback directory if no other source is configured.
        env_var:
            Environment variable name to check.
        create_if_missing:
            If True, automatically create the directory hierarchy.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        pathlib.Path:
            Resolved experiment root directory.
    '''
    if exp_root is not None:
        target = pathlib.Path(exp_root)
    else:
        widgets = widgets_mod.NotebookWidgets(dbutils=dbutils)
        if widgets.has_dbutils:
            try:
                target = widgets.path('exp_root', default=default)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                target = pathlib.Path(default)
        else:
            env_val = os.environ.get(env_var) or os.environ.get('EXP_ROOT')
            if env_val:
                target = pathlib.Path(env_val)
            else:
                target = pathlib.Path(default)

    target = target.expanduser()
    if create_if_missing:
        target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_notebook_io(
    exp_root: str | pathlib.Path | None = None,
    input_dir: str | pathlib.Path | None = None,
    *,
    widgets: widgets_mod.NotebookWidgets | None = None,
    create_dirs: bool = False,
) -> NotebookIOConfig:
    '''
    Resolve unified notebook input and output directory layout.

    Establishes standard folder relationships for artifacts, results,
    and raw input datasets relative to the resolved experiment root.

    Args:
        exp_root:
            Optional explicit experiment root path.
        input_dir:
            Optional explicit raw input data directory.
        widgets:
            Optional existing `NotebookWidgets` instance.
        create_dirs:
            If True, automatically create the resolved directories.

    Returns:
        NotebookIOConfig:
            Container with resolved experiment, artifact, and paths.
    '''
    active_widgets = (
        widgets if widgets is not None
        else widgets_mod.NotebookWidgets()
    )
    if exp_root is not None:
        resolved_root = pathlib.Path(exp_root).expanduser()
    else:
        resolved_root = active_widgets.path(
            'exp_root',
            default='./experiment',
            label='Experiment Root',
            create_if_missing=create_dirs,
        )

    artifacts_dir = resolved_root / 'artifacts'
    results_dir = resolved_root / 'results'

    if input_dir is not None:
        resolved_input = pathlib.Path(input_dir).expanduser()
    else:
        default_input = resolved_root / 'input'
        resolved_input = active_widgets.path(
            'input_dir',
            default=default_input,
            label='Raw Data Input Directory',
            create_if_missing=create_dirs,
        )

    if create_dirs:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    return NotebookIOConfig(
        exp_root=resolved_root,
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        input_dir=resolved_input,
    )


def resolve_optuna_storage(
    storage: str | None = None,
    default: str | None = None,
    *,
    widget_name: str = 'optuna_storage',
    dbutils: typing.Any = None,
) -> str:
    '''
    Resolve safe Optuna storage backend URI across runtimes.

    SQLite on shared cloud storage (DBFS or Unity Catalog Volumes)
    lacks robust POSIX file locking and can corrupt trial databases.
    When running on Databricks without an explicit remote RDBMS URI,
    this helper automatically routes SQLite to driver storage.

    Args:
        storage:
            Optional explicit storage connection string.
        default:
            Optional custom default storage string.
        widget_name:
            Widget parameter name to inspect when in Databricks.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        str:
            Environment-safe Optuna storage connection URI.
    '''
    if storage is not None:
        return storage

    widgets = widgets_mod.NotebookWidgets(dbutils=dbutils)
    if default is not None:
        fallback = default
    elif env.is_databricks():
        fallback = 'sqlite:////tmp/optuna.db'
    else:
        fallback = 'sqlite:///optuna.db'

    if widgets.has_dbutils:
        try:
            return widgets.text(widget_name, default=fallback)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            return fallback

    env_storage = os.environ.get('LANDSEG_OPTUNA_STORAGE')
    if env_storage:
        return env_storage

    return fallback

