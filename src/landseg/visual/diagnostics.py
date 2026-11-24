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
Diagnostic visual plotting utilities for model evaluation.

Provides publication-quality confusion matrix heatmaps and per-class
metric bar charts for deep learning evaluation auditing.

Public APIs:
    - `plot_confusion_matrix`: Render annotated confusion matrix heatmap.
    - `plot_per_class_metrics`: Render per-class performance bar chart.
'''

# standard imports
import pathlib
import typing
# third-party imports
import matplotlib
import matplotlib.pyplot
import numpy
import torch
# local imports
import landseg.runtime.platform as platform

if not platform.is_notebook():
    matplotlib.use('Agg')


# ----- public functions
def plot_confusion_matrix(
    matrix: numpy.ndarray | torch.Tensor,
    class_names: list[str] | None = None,
    *,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
    cmap: str = 'Blues',
) -> matplotlib.figure.Figure:
    '''
    Render an annotated confusion matrix heatmap.

    Args:
        matrix:
            2D square confusion matrix of shape [N, N].
        class_names:
            Optional list of class labels for tick markings.
        normalize:
            Whether to normalize each row by row sums to display
            percentages.
        title:
            Super-title for the figure.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure interactively.
        cmap:
            Matplotlib colormap identifier.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.

    Raises:
        ValueError:
            If `matrix` is not a 2D square matrix.
    '''
    if isinstance(matrix, torch.Tensor):
        cm = matrix.detach().cpu().to(torch.float64).numpy()
    else:
        cm = numpy.asarray(matrix, dtype=numpy.float64)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f'Confusion matrix must be a 2D square matrix, got shape {cm.shape}'
        )

    n_classes = cm.shape[0]
    display_cm = cm.copy()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_cm = numpy.divide(
            cm,
            row_sums,
            out=numpy.zeros_like(cm),
            where=row_sums != 0,
        )

    figsize = (max(6, n_classes * 0.8), max(5, n_classes * 0.7))
    fig, ax = matplotlib.pyplot.subplots(figsize=figsize)

    im = ax.imshow(
        display_cm,
        interpolation='nearest',
        cmap=cmap,
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None,
    )
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = (
        class_names
        if class_names and len(class_names) == n_classes
        else [str(i) for i in range(n_classes)]
    )

    ax.set(
        xticks=numpy.arange(n_classes),
        yticks=numpy.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel='True Class',
        xlabel='Predicted Class',
    )
    ax.set_ylim(n_classes - 0.5, -0.5)

    matplotlib.pyplot.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor',
    )

    # annotate cells with values
    thresh = display_cm.max() / 2.0 if display_cm.size > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = display_cm[i, j]
            text = f'{val * 100:.1f}%' if normalize else f'{int(cm[i, j])}'
            color = 'white' if val > thresh else 'black'
            ax.text(
                j,
                i,
                text,
                ha='center',
                va='center',
                color=color,
                fontsize=9,
            )

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

    return fig


def plot_per_class_metrics(
    metrics: dict[str, float] | numpy.ndarray | torch.Tensor,
    *,
    metric_name: str = 'IoU',
    class_names: list[str] | None = None,
    title: str | None = None,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
    bar_color: str = '#2E8B57',
) -> matplotlib.figure.Figure:
    '''
    Render a horizontal bar chart of per-class evaluation scores.

    Args:
        metrics:
            Dictionary mapping class name/index to score, or array/tensor.
        metric_name:
            Display name of the metric (e.g. 'IoU', 'F1').
        class_names:
            Optional human-readable class labels.
        title:
            Super-title for the chart.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure interactively.
        bar_color:
            Hex or named color for bars.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    if isinstance(metrics, (torch.Tensor, numpy.ndarray)):
        arr = (
            metrics.detach().cpu().numpy()
            if isinstance(metrics, torch.Tensor)
            else numpy.asarray(metrics)
        )
        labels = (
            class_names
            if class_names and len(class_names) == len(arr)
            else [f'Class {i}' for i in range(len(arr))]
        )
        values = [float(v) for v in arr]
    elif isinstance(metrics, dict):
        labels = [str(k) for k in metrics.keys()]
        values = [float(v) for v in metrics.values()]
        if class_names and len(class_names) == len(labels):
            labels = class_names
    else:
        raise TypeError(f'Unsupported metrics type: {type(metrics)}')

    n_items = len(labels)
    figsize = (8, max(4, n_items * 0.4))
    fig, ax = matplotlib.pyplot.subplots(figsize=figsize)

    y_pos = numpy.arange(n_items)
    bars = ax.barh(y_pos, values, color=bar_color, alpha=0.85, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top-down order
    ax.set_xlabel(f'Mean {metric_name}', fontsize=10, fontweight='bold')
    ax.set_xlim(0.0, max(1.0, max(values) + 0.1 if values else 1.0))

    if values:
        mean_val = float(numpy.mean(values))
        ax.axvline(
            mean_val,
            color='#D9534F',
            linestyle='--',
            linewidth=1.5,
            label=f'Mean: {mean_val:.3f}',
        )
        ax.legend(loc='lower right')

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}',
            va='center',
            fontsize=9,
        )

    ax.grid(axis='x', linestyle=':', alpha=0.6)
    chart_title = title or f'Per-Class {metric_name} Breakdown'
    ax.set_title(chart_title, fontsize=12, fontweight='bold')

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

    return fig
