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
Training loss and validation metric curve plotting utilities.
'''

# standard imports
import json
import pathlib
import typing
# third-party imports
import matplotlib
import matplotlib.pyplot
import numpy
# local imports
import landseg.runtime.platform as platform


if not platform.is_notebook():
    matplotlib.use('Agg')


# ----- public functions
def plot_training_curves(
    step_results: str | pathlib.Path | list[dict[str, typing.Any]],
    *,
    metric_name: str | None = None,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Plot training loss and validation metric trajectories across epochs.

    Renders a side-by-side visualization displaying:
    - Left: Total training loss vs. validation loss across epochs.
    - Right: Validation tracking metrics (mIoU) for all active heads.

    Args:
        step_results:
            Path to `step_results.json`, directory containing it, or
            pre-loaded list of session step dictionaries.
        metric_name:
            Optional fallback display name for the tracking metric.
        save_path:
            Optional file path to persist figure (e.g. `.png`).
        show:
            Whether to display figure inline.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    steps = _load_step_results(step_results)
    if not steps:
        raise ValueError('No training step records found to plot.')

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float | None] = []

    # Map of head_name -> list of (epoch, mean_iou)
    head_curves: dict[str, list[tuple[int, float]]] = {}
    fallback_metrics: list[tuple[int, float | None]] = []
    fallback_metric_label = metric_name

    for idx, step in enumerate(steps):
        epoch = step.get('epoch_in_phase') or step.get('global_epoch') or (idx + 1)
        epochs.append(int(epoch))

        # Training loss extraction
        train_data = step.get('training') or {}
        train_loss = train_data.get('total_objective', 0.0)
        train_losses.append(float(train_loss))

        # Validation total loss extraction
        val_data = step.get('validation') or {}
        mtl = val_data.get('mtl_metrics') or {}
        val_loss = mtl.get('total_loss') if 'total_loss' in mtl else mtl.get('loss')
        val_losses.append(float(val_loss) if val_loss is not None else None)

        # Multi-head validation metric extraction
        head_metrics = val_data.get('head_metrics') or {}
        if isinstance(head_metrics, dict) and head_metrics:
            for head_name, h_dict in head_metrics.items():
                if isinstance(h_dict, dict) and 'mean' in h_dict and h_dict['mean'] is not None:
                    head_curves.setdefault(head_name, []).append(
                        (int(epoch), float(h_dict['mean']))
                    )

        # Fallback single validation metric
        if fallback_metric_label is None:
            fallback_metric_label = step.get('validation_metrics_name') or 'Validation Metric'
        val_val = step.get('validation_metrics_value')
        fallback_metrics.append((int(epoch), float(val_val) if val_val is not None else None))

    fig, (ax_loss, ax_metric) = matplotlib.pyplot.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Loss Trajectory
    ax_loss.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2)
    valid_val_losses = [(e, l) for e, l in zip(epochs, val_losses) if l is not None]
    if valid_val_losses:
        ve, vl = zip(*valid_val_losses)
        ax_loss.plot(ve, vl, label='Val Loss', marker='s', linestyle='--', linewidth=2)

    ax_loss.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Objective Value')
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='best')

    # Panel 2: Multi-Head Evaluation Metrics
    if head_curves:
        # Plot every active head on the same metric plot
        cmap = matplotlib.pyplot.colormaps.get_cmap('tab10')
        for i, (head_name, points) in enumerate(head_curves.items()):
            valid_pts = [(e, m) for e, m in points if m is not None]
            if valid_pts:
                he, hm = zip(*valid_pts)
                color = cmap(i % 10)
                ax_metric.plot(
                    he, hm, label=f'{head_name} (mIoU)', color=color, marker='o', linewidth=2
                )
                best_idx = int(numpy.argmax(hm))
                best_epoch = he[best_idx]
                best_val = hm[best_idx]
                ax_metric.scatter(
                    [best_epoch],
                    [best_val],
                    color=color,
                    s=90,
                    zorder=5,
                )
        ax_metric.set_title(
            'Validation Metrics (All Active Heads)', fontsize=12, fontweight='bold'
        )
    else:
        # Fallback to single tracking metric if no head_metrics dictionary present
        valid_fb = [(e, m) for e, m in fallback_metrics if m is not None]
        if valid_fb:
            me, mm = zip(*valid_fb)
            ax_metric.plot(
                me, mm, label=fallback_metric_label, color='forestgreen', marker='o', linewidth=2
            )
            best_idx = int(numpy.argmax(mm))
            best_epoch = me[best_idx]
            best_metric = mm[best_idx]
            ax_metric.scatter(
                [best_epoch],
                [best_metric],
                color='crimson',
                s=120,
                zorder=5,
                label=f'Best: {best_metric:.4f} (Ep {best_epoch})',
            )
        ax_metric.set_title(
            f'Evaluation Metric ({fallback_metric_label})', fontsize=12, fontweight='bold'
        )

    ax_metric.set_xlabel('Epoch')
    ax_metric.set_ylabel('mIoU / Score')
    ax_metric.grid(True, linestyle=':', alpha=0.6)
    ax_metric.legend(loc='best')

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
    source: str | pathlib.Path | list[dict[str, typing.Any]],
) -> list[dict[str, typing.Any]]:
    '''Load session step results from list or filesystem path.'''
    if isinstance(source, list):
        return source

    path = pathlib.Path(source)
    if path.is_file():
        target = path
    elif path.is_dir():
        # Search for step_results.json in directory or session artifacts
        candidates = list(path.glob('**/step_results.json'))
        if candidates:
            target = max(candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f'Could not find step_results.json under {path}')
    else:
        raise FileNotFoundError(f'Specified path does not exist: {path}')

    with open(target, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'steps' in data and isinstance(data['steps'], list):
            return data['steps']
        return [data]
    return []
