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

Exposes interactive notebook widgets, display utilities, in-cell visual
inspection previewers, and interruption-safe training session management.

Public APIs:
    - `MissingRequiredInputError`: Error raised on missing notebook parameters.
    - `NotebookWidgets`: Ergonomic widget wrapper across runtimes.
    - `is_notebook`: Probe whether code is running inside a notebook.
    - `plot_qualitative_predictions`: Side-by-side segmentation plots.
    - `plot_training_curves`: In-cell training loss and metric curves.
    - `render_prediction_preview`: Automated prediction inspection preview.
    - `require_widget`: Enforce and validate required notebook widgets.
    - `safe_training_session`: Interruption-safe training context manager.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'MissingRequiredInputError',
    'NotebookWidgets',
    # functions
    'is_notebook',
    'plot_qualitative_predictions',
    'plot_training_curves',
    'render_prediction_preview',
    'require_widget',
    'safe_training_session',
]


# for static check
if typing.TYPE_CHECKING:
    from .env import (
        is_notebook,
    )
    from .telemetry import (
        plot_qualitative_predictions,
        plot_training_curves,
        render_prediction_preview,
        safe_training_session,
    )
    from .widgets import (
        MissingRequiredInputError,
        NotebookWidgets,
        require_widget,
    )


def __getattr__(name: str):
    if name in {'MissingRequiredInputError', 'NotebookWidgets', 'require_widget'}:
        obj = importlib.import_module('.widgets', __package__)
        return getattr(obj, name)

    if name in {
        'plot_qualitative_predictions',
        'plot_training_curves',
        'render_prediction_preview',
        'safe_training_session',
    }:
        obj = importlib.import_module('.telemetry', __package__)
        return getattr(obj, name)

    if name in {'is_notebook'}:
        obj = importlib.import_module('.env', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
