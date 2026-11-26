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

'''Training loss and validation metric curve plotting utilities.'''

# standard imports
import dataclasses
import pathlib
import typing
# third-party imports
import matplotlib
import matplotlib.axes
import matplotlib.pyplot
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.runtime as runtime


if not runtime.is_notebook():
    matplotlib.use('Agg')


# ----- private dataclasses
@dataclasses.dataclass(frozen=True)
class _CurveStepPoint:
    '''Normalized loss and metric values for an epoch step.'''
    epoch: int
    train_loss: float
    head_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    metric_label: str | None = None


# ----- public functions
def plot_training_curves(
    step_results: str | pathlib.Path,
    *,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Plot training loss and validation metric trajectories across epochs.

    Renders a side-by-side visualization displaying:
    - Left: Training loss trajectory across epochs.
    - Right: Validation tracking metrics (mIoU) for all active heads.

    Args:
        step_results:
            Path to `step_results.json`.
        save_path:
            Optional file path to persist figure (e.g. `.png`).
        show:
            Whether to display figure inline.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    points = _load_step_results(step_results)

    # map of total loss over epoch
    loss_curve: dict[int, float] = {}
    # map of head_name -> list of (epoch, mean_iou)
    head_curves: dict[str, list[tuple[int, float]]] = {}

    for p in points:
        loss_curve[p.epoch] = p.train_loss

        for head_name, mean_val in p.head_metrics.items():
            head_curves.setdefault(head_name, []).append((p.epoch, mean_val))

    fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(14, 5))

    # panel 1: Training Loss Trajectory
    _render_loss_curve(ax1, loss_curve)
    # panel 2: Multi-Head Evaluation Metrics
    _render_head_curves(ax2, head_curves)

    matplotlib.pyplot.tight_layout()

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()

    return fig


# ----- private helpers
def _load_step_results(
    source: str | pathlib.Path,
) -> list[_CurveStepPoint]:
    '''Load session step results from file into typed point data.'''
    path = pathlib.Path(source)
    if path.is_file():
        target = path
    elif path.is_dir():
        # search for step_results.json in directory or session artifacts
        candidates = list(path.glob('**/step_results.json'))
        if candidates:
            target = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f'No step_results.json under {path}')
    else:
        raise FileNotFoundError(f'Specified path does not exist: {path}')

    steps = artifacts.Controller[list[dict[str, typing.Any]]](target).fetch()
    if not steps:
        raise ValueError('No training step records found to plot.')

    points: list[_CurveStepPoint] = []
    for index, step in enumerate(steps):

        epoch = step.get('epoch_in_phase') or (index + 1)
        train_data = step.get('training') or {}
        val_data = step.get('validation') or {}

        head_metrics = {}
        for head, val in (val_data.get('head_metrics') or {}).items():
            if (
                isinstance(val, dict)
                and 'mean' in val
                and val['mean'] is not None
            ):
                head_metrics[head] = float(val['mean'])

        p = _CurveStepPoint(
            epoch=int(epoch),
            train_loss=float(train_data.get('total_objective', 0.0)),
            head_metrics=head_metrics,
            metric_label=step.get('validation_metrics_name'),
        )
        points.append(p)

    return points


def _render_loss_curve(
    ax: matplotlib.axes.Axes,
    loss_curve: dict[int, float]
) -> None:
    '''Render training loss trajectories on target axis.'''
    ax.plot(
        list(loss_curve.keys()),
        list(loss_curve.values()),
        label='Train Loss',
        marker='o',
        linewidth=2,
        color='tab:blue',
    )
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Objective Value')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')


def _render_head_curves(
    ax: matplotlib.axes.Axes,
    head_curves: dict[str, list[tuple[int, float]]],
) -> None:
    '''Render multi-head validation metric curves on target axis.'''
    if not head_curves:
        ax.text(
            0.5,
            0.5,
            'No validation metrics recorded for plotting',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            color='gray',
        )
        return

    cmap = matplotlib.pyplot.colormaps.get_cmap('tab10')
    for i, (head_name, pts) in enumerate(head_curves.items()):
        valid_pts = [(e, m) for e, m in pts if m is not None]
        if not valid_pts:
            continue
        he, hm = zip(*valid_pts)
        color = cmap(i % 10)
        ax.plot(
            he,
            hm,
            label=f'{head_name} (mIoU)',
            color=color,
            marker='o',
            linewidth=2,
        )
        best_idx = int(numpy.argmax(hm))
        ax.scatter([he[best_idx]], [hm[best_idx]], color=color, s=90, zorder=5)

    ax.set_title(
        'Validation Metrics (All Active Heads)',
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU / Score')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')
