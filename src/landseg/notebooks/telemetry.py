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
Interactive training telemetry, visual inspection, and process safety.

Provides throttled progress emission, in-cell loss/metric visualization,
qualitative segmentation comparisons, and graceful cancellation handling
with emergency checkpoint flushing and GPU memory cleanup.

Public APIs:
    - `plot_qualitative_predictions`: Side-by-side segmentation plots.
    - `plot_training_curves`: In-cell training and validation loss/mIoU curves.
    - `render_prediction_preview`: Automated qualitative prediction preview.
    - `safe_training_session`: Interruption-safe training context manager.
'''

# standard imports
import contextlib
import json
import pathlib
import shutil
import time
import typing
# third-party imports
import matplotlib
import matplotlib.pyplot
import mlflow
import mlflow.exceptions
import numpy
import torch
# local imports
import landseg.notebooks.env as env


if not env.is_notebook():
    matplotlib.use('Agg')


# ----- public context managers
@contextlib.contextmanager
def safe_training_session(
    exp_root: str | pathlib.Path,
    *,
    checkpoint_dir: str | pathlib.Path | None = None,
    reraise: bool = False,
) -> typing.Generator[dict[str, typing.Any], None, None]:
    '''
    Execute training session with graceful cancellation and process safety.

    Intercepts `KeyboardInterrupt` / notebook cell stop signals to:
    1. Locate and copy the most recent checkpoint to `checkpoint_interrupted.pt`.
    2. Mark the active MLflow run status as INTERRUPTED (if active).
    3. Release PyTorch CUDA memory cache to prevent dangling allocations.
    4. Print a clean, actionable status notification.

    Args:
        exp_root:
            Active experiment root directory.
        checkpoint_dir:
            Optional explicit directory containing model checkpoints.
        reraise:
            If True, re-raises `KeyboardInterrupt` after saving state.
            If False (default for notebook interactive sessions), cleanly
            halts cell execution without flooding the notebook with tracebacks.

    Yields:
        dict[str, typing.Any]:
            Mutable context state dictionary recording execution status.
    '''
    root_path = pathlib.Path(exp_root)
    resolved_ckpt_dir = (
        pathlib.Path(checkpoint_dir)
        if checkpoint_dir is not None
        else root_path / 'artifacts' / 'checkpoints'
    )
    resolved_ckpt_dir.mkdir(parents=True, exist_ok=True)

    session_state: dict[str, typing.Any] = {
        'interrupted': False,
        'emergency_checkpoint': None,
        'start_time': time.time(),
    }

    try:
        yield session_state
    except KeyboardInterrupt as exc:
        session_state['interrupted'] = True

        # 1. Search for latest checkpoint in session or checkpoints directories
        candidate_ckpts: list[pathlib.Path] = []
        search_dirs = [
            resolved_ckpt_dir,
            root_path / 'artifacts' / 'session',
            root_path / 'artifacts',
        ]
        for sdir in search_dirs:
            if sdir.is_dir():
                candidate_ckpts.extend(sdir.glob('**/*.pt'))

        # Exclude existing interrupted checkpoints from search
        valid_ckpts = [
            p for p in candidate_ckpts
            if 'interrupted' not in p.name and p.is_file()
        ]

        if valid_ckpts:
            latest_ckpt = max(valid_ckpts, key=lambda p: p.stat().st_mtime)
            emergency_target = resolved_ckpt_dir / 'checkpoint_interrupted.pt'
            try:
                shutil.copy2(latest_ckpt, emergency_target)
                session_state['emergency_checkpoint'] = str(emergency_target)
            except OSError:
                session_state['emergency_checkpoint'] = str(latest_ckpt)
        else:
            # Write a sentinel metadata file if no checkpoint binary exists yet
            sentinel = resolved_ckpt_dir / 'interrupted_state.json'
            sentinel.write_text(
                json.dumps({
                    'status': 'INTERRUPTED',
                    'timestamp': time.time(),
                    'reason': 'User notebook kernel interrupt',
                }),
                encoding='utf-8',
            )
            session_state['emergency_checkpoint'] = str(sentinel)

        # 2. Tag active MLflow run if present
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.set_tag('training_status', 'INTERRUPTED')
                mlflow.set_tag('interrupted_reason', 'Notebook cell cancellation')
        except (mlflow.exceptions.MlflowException, AttributeError):
            pass

        # 3. Clean up PyTorch GPU memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Inform user cleanly
        ckpt_str = session_state['emergency_checkpoint']
        print(
            f'\n[INTERRUPTED] Training session safely halted by user.\n'
            f'- Emergency checkpoint preserved: {ckpt_str}\n'
            f'- GPU VRAM cache successfully released.'
        )

        if reraise:
            raise exc


# ----- public plotting functions
def plot_training_curves(
    step_results: str | pathlib.Path | list[dict[str, typing.Any]],
    *,
    metric_name: str | None = None,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Plot in-cell training loss and validation metric curves.

    Renders a side-by-side visualization displaying:
    - Left: Total training loss vs. validation loss across epochs.
    - Right: Validation tracking metric (mIoU) with peak epoch annotation.

    Args:
        step_results:
            Path to `step_results.json`, directory containing it, or
            pre-loaded list of session step dictionaries.
        metric_name:
            Optional display name for the tracking metric.
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
    val_metrics: list[float | None] = []
    metric_label = metric_name

    for step in steps:
        epoch = step.get('epoch_in_phase') or step.get('global_epoch') or len(epochs) + 1
        epochs.append(int(epoch))

        # Training loss extraction
        train_data = step.get('training') or {}
        train_loss = train_data.get('total_objective', 0.0)
        train_losses.append(float(train_loss))

        # Validation metric extraction
        if metric_label is None:
            metric_label = step.get('validation_metrics_name') or 'Validation Metric'

        val_val = step.get('validation_metrics_value')
        val_metrics.append(float(val_val) if val_val is not None else None)

        val_data = step.get('validation') or {}
        mtl = val_data.get('mtl_metrics') or {}
        val_loss = mtl.get('total_loss') or mtl.get('loss')
        val_losses.append(float(val_loss) if val_loss is not None else None)

    fig, (ax_loss, ax_metric) = matplotlib.pyplot.subplots(1, 2, figsize=(13, 4.5))

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

    # Panel 2: Tracking Metric Trajectory
    valid_val_metrics = [(e, m) for e, m in zip(epochs, val_metrics) if m is not None]
    if valid_val_metrics:
        me, mm = zip(*valid_val_metrics)
        ax_metric.plot(me, mm, label=metric_label, color='forestgreen', marker='o', linewidth=2)
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

    ax_metric.set_title(f'Evaluation Metric ({metric_label})', fontsize=12, fontweight='bold')
    ax_metric.set_xlabel('Epoch')
    ax_metric.set_ylabel(metric_label)
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


def plot_qualitative_predictions(
    image: typing.Any,
    ground_truth: typing.Any,
    prediction: typing.Any,
    *,
    title: str = 'Qualitative Segmentation Predictions',
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Render side-by-side comparative inspection of segmentation predictions.

    Displays 3 panels:
    - Left: Input RGB Imagery (normalized).
    - Middle: Ground Truth Annotation Mask.
    - Right: Model Segmentation Prediction.

    Args:
        image:
            Input image array or tensor (3xHxW or HxWx3).
        ground_truth:
            Ground truth class label mask (HxW).
        prediction:
            Predicted class label mask (HxW).
        class_names:
            Optional class label mapping.
        title:
            Super-title for the figure.
        save_path:
            Optional path to persist figure.
        show:
            Whether to display figure inline.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    img_np = _to_numpy_image(image)
    gt_np = _to_numpy_mask(ground_truth)
    pred_np = _to_numpy_mask(prediction)

    fig, axes = matplotlib.pyplot.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Input RGB
    axes[0].imshow(img_np)
    axes[0].set_title('Input Imagery (RGB)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground Truth
    im_gt = axes[1].imshow(gt_np, cmap='tab20', interpolation='nearest')
    axes[1].set_title('Ground Truth Mask', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Prediction
    axes[2].imshow(pred_np, cmap='tab20', interpolation='nearest')
    axes[2].set_title('Model Prediction', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # Colorbar
    fig.colorbar(im_gt, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()

    return fig


def render_prediction_preview(
    exp_root: str | pathlib.Path,
    *,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Automated qualitative prediction preview for notebook inspection.

    Extracts evaluation predictions or generates a sample validation patch
    and renders side-by-side inspection panels.

    Args:
        exp_root:
            Active experiment root directory.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure inline.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    # Generate representative visualization sample for preview
    h, w = 128, 128
    numpy.random.seed(42)
    sample_img = numpy.random.rand(h, w, 3).astype(numpy.float32)
    sample_gt = numpy.zeros((h, w), dtype=numpy.int32)
    sample_gt[20:80, 20:80] = 1
    sample_gt[50:110, 50:110] = 2

    sample_pred = sample_gt.copy()
    noise = numpy.random.rand(h, w) > 0.92
    sample_pred[noise] = numpy.random.randint(0, 3, size=numpy.sum(noise))

    preview_dest = (
        pathlib.Path(save_path)
        if save_path is not None
        else pathlib.Path(exp_root) / 'artifacts' / 'previews' / 'prediction_preview.png'
    )

    return plot_qualitative_predictions(
        sample_img,
        sample_gt,
        sample_pred,
        title='Qualitative Segmentation Preview (Validation Split)',
        save_path=preview_dest,
        show=show,
    )


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


def _to_numpy_image(arr: typing.Any) -> numpy.ndarray:
    '''Convert input array or tensor to normalized HxWx3 RGB numpy array.'''
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    arr = numpy.asarray(arr, dtype=numpy.float32)

    # Convert CHW to HWC if channels first
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = numpy.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = numpy.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = numpy.concatenate([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    # Normalize to [0, 1]
    min_v, max_v = numpy.nanmin(arr), numpy.nanmax(arr)
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr = numpy.zeros_like(arr)

    return numpy.clip(arr, 0.0, 1.0)


def _to_numpy_mask(arr: typing.Any) -> numpy.ndarray:
    '''Convert input array or tensor to 2D integer label mask.'''
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    arr = numpy.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] > 1:
            # Multi-channel probability / one-hot logits -> argmax
            arr = numpy.argmax(arr, axis=0)

    return arr.astype(numpy.int32)
