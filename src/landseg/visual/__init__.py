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
Top-level namespace for `landseg.visual`.

Provides pure visualization utilities for metric plotting, segmentation
prediction previews, palette colorization, and multi-head qualitative
inspections across CLI and interactive notebook environments.

Public APIs:
    - `DEFAULT_ERROR_PALETTE`: Standard orange/gray error highlight palette.
    - `DEFAULT_MATCH_PALETTE`: Standard green/gray match highlight palette.
    - `FORESTRY_FRI_PALETTE`: Standard Ontario FRI tree species palette.
    - `NLCD_16_PALETTE`: Standard 16-class National Land Cover Database palette.
    - `colorize`: Map 2D class integer masks to 3D RGB tensors.
    - `colorize_continuous`: Map 2D float tensors to 3D RGB colormaps.
    - `generate_prediction_previews`: Produce inferred preview figures per head.
    - `plot_qualitative_predictions`: Side-by-side segmentation inspection.
    - `plot_training_curves`: Plot multi-head training and validation curves.
    - `render_prediction_preview`: Automated prediction preview display.
    - `select_representative_block`: Deterministically select a diverse block.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # constants
    'DEFAULT_ERROR_PALETTE',
    'DEFAULT_MATCH_PALETTE',
    'FORESTRY_FRI_PALETTE',
    'NLCD_16_PALETTE',
    # functions
    'colorize',
    'colorize_continuous',
    'generate_prediction_previews',
    'plot_confusion_matrix',
    'plot_per_class_metrics',
    'plot_qualitative_predictions',
    'plot_training_curves',
    'render_prediction_preview',
    'select_representative_block',
]


# for static check
if typing.TYPE_CHECKING:
    from .color import (
        DEFAULT_ERROR_PALETTE,
        DEFAULT_MATCH_PALETTE,
        FORESTRY_FRI_PALETTE,
        NLCD_16_PALETTE,
        colorize,
        colorize_continuous,
    )
    from .curves import (
        plot_training_curves,
    )
    from .diagnostics import (
        plot_confusion_matrix,
        plot_per_class_metrics,
    )
    from .preview import (
        generate_prediction_previews,
        plot_qualitative_predictions,
        render_prediction_preview,
        select_representative_block,
    )


def __getattr__(name: str):
    if name in {
        'DEFAULT_ERROR_PALETTE',
        'DEFAULT_MATCH_PALETTE',
        'FORESTRY_FRI_PALETTE',
        'NLCD_16_PALETTE',
        'colorize',
        'colorize_continuous',
    }:
        obj = importlib.import_module('.color', __package__)
        return getattr(obj, name)

    if name in {'plot_training_curves'}:
        obj = importlib.import_module('.curves', __package__)
        return getattr(obj, name)

    if name in {
        'plot_confusion_matrix',
        'plot_per_class_metrics',
    }:
        obj = importlib.import_module('.diagnostics', __package__)
        return getattr(obj, name)

    if name in {
        'generate_prediction_previews',
        'plot_qualitative_predictions',
        'render_prediction_preview',
        'select_representative_block',
    }:
        obj = importlib.import_module('.preview', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
