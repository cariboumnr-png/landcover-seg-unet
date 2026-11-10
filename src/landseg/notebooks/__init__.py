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
    - `ThrottledProgress`: Console emission throttler.
    - `is_databricks`: Probe whether runtime is inside Databricks.
    - `is_notebook`: Probe whether code is running inside a notebook.
    - `load_study_trials`: Load persisted sweep trials dataframe.
    - `plot_qualitative_predictions`: Side-by-side segmentation plots.
    - `plot_training_curves`: In-cell training loss and metric curves.
    - `register_unity_catalog_model`: Register checkpoint to Unity Catalog.
    - `render_prediction_preview`: Automated prediction inspection preview.
    - `resolve_exp_root`: Resolve experiment root path across contexts.
    - `resolve_notebook_io`: Resolve unified notebook I/O structure.
    - `resolve_optuna_storage`: Resolve environment-safe Optuna URI.
    - `safe_training_session`: Interruption-safe training context manager.
    - `sync_optuna_study`: Persist Optuna study trials to volume storage.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'NotebookIOConfig',
    'NotebookWidgets',
    'ThrottledProgress',
    # functions
    'is_databricks',
    'is_notebook',
    'load_study_trials',
    'plot_qualitative_predictions',
    'plot_training_curves',
    'register_unity_catalog_model',
    'render_prediction_preview',
    'resolve_exp_root',
    'resolve_notebook_io',
    'resolve_optuna_storage',
    'safe_training_session',
    'sync_optuna_study',
]


# for static check
if typing.TYPE_CHECKING:
    from .env import (
        is_databricks,
        is_notebook,
    )
    from .io import (
        NotebookIOConfig,
        load_study_trials,
        resolve_exp_root,
        resolve_notebook_io,
        resolve_optuna_storage,
        sync_optuna_study,
    )
    from .registry import (
        register_unity_catalog_model,
    )
    from .telemetry import (
        ThrottledProgress,
        plot_qualitative_predictions,
        plot_training_curves,
        render_prediction_preview,
        safe_training_session,
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
        'load_study_trials',
        'resolve_exp_root',
        'resolve_notebook_io',
        'resolve_optuna_storage',
        'sync_optuna_study',
    }:
        obj = importlib.import_module('.io', __package__)
        return getattr(obj, name)

    if name in {
        'ThrottledProgress',
        'plot_qualitative_predictions',
        'plot_training_curves',
        'render_prediction_preview',
        'safe_training_session',
    }:
        obj = importlib.import_module('.telemetry', __package__)
        return getattr(obj, name)

    if name in {'register_unity_catalog_model'}:
        obj = importlib.import_module('.registry', __package__)
        return getattr(obj, name)

    if name in {'is_databricks', 'is_notebook'}:
        obj = importlib.import_module('.env', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
