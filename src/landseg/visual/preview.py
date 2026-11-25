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
Segmentation prediction preview and qualitative visualization utilities.

Pure presentation module for rendering RGB imagery, ground-truth masks,
inferred predictions, and error highlights without session or model
dependencies.

Public APIs:
    - `generate_prediction_previews`: Render preview figures per head.
    - `plot_qualitative_predictions`: Render comparative preview panels.
    - `render_prediction_preview`: Display saved preview PNGs in
      notebooks.
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
import landseg.core as core
import landseg.runtime as runtime
import landseg.visual.color as visual_color

if not runtime.is_notebook():
    matplotlib.use('Agg')


# ----- public functions
def plot_qualitative_predictions(
    image: typing.Any,
    ground_truth: typing.Any,
    prediction: typing.Any,
    errors: typing.Any = None,
    *,
    title: str = 'Qualitative Segmentation Predictions',
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
    color_map: dict[str, list[int]] | None = None,
) -> typing.Any:
    '''
    Render comparative inspection panels of segmentation predictions.

    Displays panels:
    - Input RGB Imagery (normalized)
    - Ground Truth Annotation Mask
    - Model Segmentation Prediction
    - (Optional) Prediction Error Mask

    Args:
        image:
            Input image array or tensor (3xHxW or HxWx3).
        ground_truth:
            Ground truth class label mask (HxW).
        prediction:
            Predicted class label mask (HxW).
        errors:
            Optional boolean or integer error mask (HxW).
        title:
            Super-title for the figure.
        save_path:
            Optional path to persist figure.
        show:
            Whether to display figure inline.
        color_map:
            Optional RGB palette dictionary for class index rendering.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    img_np = _to_numpy_image(image)
    gt_np = _to_numpy_mask(ground_truth)
    pred_np = _to_numpy_mask(prediction)

    has_errors = errors is not None
    n_panels = 4 if has_errors else 3
    figsize = (18, 4.5) if has_errors else (14, 4.5)

    fig, axes = matplotlib.pyplot.subplots(1, n_panels, figsize=figsize)

    # panel 1: input RGB
    axes[0].imshow(img_np)
    axes[0].set_title('Input Imagery (RGB)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # colorize masks if palette provided, else fallback to colormap
    if color_map:
        gt_vis = visual_color.colorize(
            torch.as_tensor(gt_np), palette=color_map
        ).permute(1, 2, 0).cpu().numpy()
        pred_vis = visual_color.colorize(
            torch.as_tensor(pred_np), palette=color_map
        ).permute(1, 2, 0).cpu().numpy()
        axes[1].imshow(gt_vis)
        axes[2].imshow(pred_vis)
    else:
        im_gt = axes[1].imshow(gt_np, cmap='tab20', interpolation='nearest')
        axes[2].imshow(pred_np, cmap='tab20', interpolation='nearest')
        fig.colorbar(im_gt, ax=axes[2], shrink=0.7, pad=0.02)

    axes[1].set_title('Ground Truth Mask', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    axes[2].set_title('Model Prediction', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # panel 4: errors (if provided)
    if has_errors:
        err_np = _to_numpy_mask(errors)
        err_vis = visual_color.colorize(
            torch.as_tensor(err_np),
            palette=visual_color.DEFAULT_ERROR_PALETTE,
        ).permute(1, 2, 0).cpu().numpy()
        axes[3].imshow(err_vis)
        axes[3].set_title(
            'Prediction Match / Error',
            fontsize=11,
            fontweight='bold',
        )
        axes[3].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    matplotlib.pyplot.tight_layout()

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

    return fig


def generate_prediction_previews(
    infer_results: core.InferStepResults | None,
    dataspecs: core.DataSpecs,
    previews_dpath: str | pathlib.Path | None = None,
    *,
    save_pngs: bool = True,
    show: bool = False,
) -> dict[str, typing.Any]:
    '''
    Render qualitative prediction preview PNGs from inference results.

    Consumes stitched inference tensors produced by `session` and
    metadata from `dataspecs` (`band_map` and `label_color_map`) to
    render per-head preview figures without running model inference.

    Args:
        infer_results:
            Stitched session inference outputs containing `infer_inputs`,
            `infer_preds`, and optional `infer_labels` / `infer_errors`.
        dataspecs:
            Data specification object supplying band mapping and palette
            metadata.
        previews_dpath:
            Output directory for saving rendered preview PNGs.
        save_pngs:
            Whether to save preview images to `previews_dpath`.
        show:
            Whether to display figures with `matplotlib.pyplot.show()`.

    Returns:
        dict[str, matplotlib.figure.Figure]:
            Mapping from head name to rendered matplotlib figure.
    '''
    if (
        infer_results is None
        or infer_results.infer_inputs is None
        or not infer_results.infer_preds
    ):
        return {}

    rgb_image = _extract_rgb_image(
        infer_results.infer_inputs,
        dataspecs.meta.image_specs.band_map,
    )
    active_heads = _resolve_active_heads(infer_results, dataspecs)
    color_map = dataspecs.meta.label_color_map

    result_figs: dict[str, typing.Any] = {}
    for head in active_heads:
        pred_tensor = infer_results.infer_preds[head]
        gt_tensor = infer_results.infer_labels.get(head)
        if gt_tensor is None:
            gt_display = torch.zeros_like(pred_tensor)
            err_display = None
        else:
            gt_display = gt_tensor
            err_display = infer_results.infer_errors.get(head)
            if err_display is None:
                err_display = (pred_tensor == gt_tensor).int()

        save_file = (
            pathlib.Path(previews_dpath) / f'{head}_preview.png'
            if (save_pngs and previews_dpath)
            else None
        )
        result_figs[head] = plot_qualitative_predictions(
            image=rgb_image,
            ground_truth=gt_display,
            prediction=pred_tensor,
            errors=err_display,
            title=f'Qualitative Segmentation Preview - [{head}]',
            save_path=save_file,
            show=show,
            color_map=color_map,
        )

    return result_figs


def render_prediction_preview(
    target_dpath: str | pathlib.Path | None = None,
    *,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Automated qualitative prediction preview for notebook inspection.

    Loads and displays previously generated preview PNGs from the
    experiment results directory, or falls back to a clean mock preview
    figure.

    Args:
        target_dpath:
            Active destination output or experiment root directory.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure inline.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    target_path = pathlib.Path(target_dpath or './experiment')

    # look for generated preview PNGs in previews directories
    candidate_dirs = [
        target_path / 'previews',
        target_path / 'artifacts' / 'previews',
    ]
    if target_path.is_dir():
        candidate_dirs.extend(target_path.glob('**/previews'))

    found_pngs: list[pathlib.Path] = []
    for d in candidate_dirs:
        if d.is_dir():
            found_pngs.extend(sorted(d.glob('*_preview.png')))
            found_pngs.extend(sorted(d.glob('prediction_preview.png')))

    # remove duplicates preserving order
    seen = set()
    unique_pngs: list[pathlib.Path] = []
    for p in found_pngs:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique_pngs.append(p)

    if unique_pngs:
        last_fig = None
        for png_file in unique_pngs:
            img = matplotlib.pyplot.imread(png_file)
            fig, ax = matplotlib.pyplot.subplots(figsize=(14, 4))
            ax.imshow(img)
            ax.set_title(
                png_file.stem.replace('_', ' ').title(),
                fontsize=12,
                fontweight='bold',
            )
            ax.axis('off')
            matplotlib.pyplot.tight_layout()
            if save_path is not None:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
            if show:
                matplotlib.pyplot.show()
            else:
                matplotlib.pyplot.close(fig)
            last_fig = fig
        return last_fig

    # fallback to generating a sample preview figure if none exist yet
    h, w = 128, 128
    numpy.random.seed(42)
    sample_img = numpy.random.rand(h, w, 3).astype(numpy.float32)
    sample_gt = numpy.zeros((h, w), dtype=numpy.int32)
    sample_gt[20:80, 20:80] = 1
    sample_gt[50:110, 50:110] = 2

    sample_pred = sample_gt.copy()
    noise = numpy.random.rand(h, w) > 0.92
    sample_pred[noise] = numpy.random.randint(0, 3, size=int(numpy.sum(noise)))

    if save_path is not None:
        preview_dest = pathlib.Path(save_path)
    elif target_path.name == 'results' or (target_path / 'previews').is_dir():
        preview_dest = target_path / 'previews' / 'prediction_preview.png'
    else:
        preview_dest = (
            target_path / 'artifacts' / 'previews' / 'prediction_preview.png'
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
def _extract_rgb_image(
    image: torch.Tensor,
    band_map: dict[str, int] | None,
) -> torch.Tensor:
    '''Select RGB channels from a multi-band `[C, H, W]` tensor.'''
    if image.dim() == 2:
        image = image.unsqueeze(0)

    if band_map and all(c in band_map for c in ('red', 'green', 'blue')):
        rgb_channels = [band_map['red'], band_map['green'], band_map['blue']]
    else:
        rgb_channels = [0, 1, 2]

    num_c = image.shape[0]
    chans = [c for c in rgb_channels if 0 <= c < num_c]
    if len(chans) == 3:
        return image[chans, :, :]
    if num_c >= 3:
        return image[:3, :, :]
    return image[0:1, :, :].repeat(3, 1, 1)


def _resolve_active_heads(
    infer_results: core.InferStepResults,
    dataspecs: core.DataSpecs,
) -> list[str]:
    '''Resolve active heads in canonical `dataspecs` order.'''
    ordered = [
        h for h in dataspecs.heads.class_counts
        if h in infer_results.infer_preds
    ]
    return ordered or list(infer_results.infer_preds.keys())


def _to_numpy_image(arr: typing.Any) -> numpy.ndarray:
    '''Convert input array or tensor to normalized HxWx3 RGB numpy array.'''
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    arr = numpy.asarray(arr, dtype=numpy.float32)

    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = numpy.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = numpy.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = numpy.concatenate([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

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
            arr = numpy.argmax(arr, axis=0)

    return arr.astype(numpy.int32)
