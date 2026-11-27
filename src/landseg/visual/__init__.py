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
prediction previews, palette colorization, diagnostic heatmaps, spatial
layout grids, and multi-band spectral composites.

Public APIs:
    - `DEFAULT_ERROR_PALETTE`: Standard orange/gray error palette.
    - `DEFAULT_MATCH_PALETTE`: Standard green/gray match palette.
    - `FORESTRY_FRI_PALETTE`: Standard Ontario FRI tree species palette.
    - `NLCD_16_PALETTE`: Standard 16-class NLCD land cover palette.
    - `colorize`: Map 2D class integer masks to 3D RGB tensors.
    - `colorize_continuous`: Map 2D float tensors to 3D RGB colormaps.
    - `generate_prediction_previews`: Render preview figures per head.
    - `generate_training_curves`: Plot multi-head training loss and metrics.
    - `plot_class_frequencies`: Render class pixel count chart.
    - `plot_confusion_matrix`: Render normalized confusion matrix.
    - `plot_grid_layout`: Render spatial grid tiling and splits.
    - `plot_per_class_metrics`: Render per-class metric bar chart.
    - `plot_qualitative_predictions`: Render comparative preview panels.
    - `plot_spectral_composite`: Render stretched spectral composite.
    - `render_prediction_preview`: Automated prediction preview display.
    - `render_training_curves`: Render training curves.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'DEFAULT_ERROR_PALETTE',
    'DEFAULT_MATCH_PALETTE',
    'FORESTRY_FRI_PALETTE',
    'NLCD_16_PALETTE',
    'colorize',
    'colorize_continuous',
    'generate_prediction_previews',
    'generate_training_curves',
    'plot_class_frequencies',
    'plot_confusion_matrix',
    'plot_grid_layout',
    'plot_per_class_metrics',
    'plot_qualitative_predictions',
    'plot_spectral_composite',
    'render_prediction_preview',
    'render_training_curves',
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
        generate_training_curves,
        render_training_curves,
    )
    from .diagnostics import (
        plot_confusion_matrix,
        plot_per_class_metrics,
    )
    from .preview import (
        generate_prediction_previews,
        plot_qualitative_predictions,
        render_prediction_preview,
    )
    from .spatial import (
        plot_class_frequencies,
        plot_grid_layout,
        plot_spectral_composite,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.color': [
        'DEFAULT_ERROR_PALETTE',
        'DEFAULT_MATCH_PALETTE',
        'FORESTRY_FRI_PALETTE',
        'NLCD_16_PALETTE',
        'colorize',
        'colorize_continuous',
    ],
    '.curves': [
        'generate_training_curves',
        'render_training_curves',
    ],
    '.diagnostics': [
        'plot_confusion_matrix',
        'plot_per_class_metrics',
    ],
    '.preview': [
        'generate_prediction_previews',
        'plot_qualitative_predictions',
        'render_prediction_preview',
    ],
    '.spatial': [
        'plot_class_frequencies',
        'plot_grid_layout',
        'plot_spectral_composite',
    ],
})
