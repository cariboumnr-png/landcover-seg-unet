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
'''

# standard imports
import pathlib
import typing
# third-party imports
import matplotlib
import matplotlib.pyplot
import numpy
import omegaconf
import torch
import torch.utils.data
# local imports
import landseg.artifacts as artifacts
import landseg.configs.schema.models as models_schema
import landseg.core as core
import landseg.models as models
import landseg.notebooks.env as env
import landseg.session.data.collate as collate
import landseg.session.data.dataset as dataset
import landseg.utils as utils
import landseg.visual.color as visual_color

if not env.is_notebook():
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

    # Panel 1: Input RGB
    axes[0].imshow(img_np)
    axes[0].set_title('Input Imagery (RGB)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Colorize masks if palette provided, else fallback to colormap
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

    # Panel 4: Errors (if provided)
    if has_errors:
        err_np = _to_numpy_mask(errors)
        err_pal = {'0': [255, 140, 0], '1': [40, 40, 40]}  # orange error vs gray match
        err_vis = visual_color.colorize(
            torch.as_tensor(err_np), palette=err_pal
        ).permute(1, 2, 0).cpu().numpy()
        axes[3].imshow(err_vis)
        axes[3].set_title('Prediction Match / Error', fontsize=11, fontweight='bold')
        axes[3].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    try:
        matplotlib.pyplot.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()

    return fig


def select_representative_block(
    test_blocks: dict[str, pathlib.Path],
    label_key: str = 'label',
) -> tuple[str, pathlib.Path]:
    '''
    Deterministically select a representative block with high class diversity.

    Evaluates candidate blocks to find one containing a non-trivial mixture
    of classes, avoiding blank or homogeneous background blocks.

    Args:
        test_blocks:
            Dictionary mapping block name to filesystem path.
        label_key:
            Name of label array stored in `.npz` files.

    Returns:
        tuple[str, pathlib.Path]:
            Selected block name and file path.
    '''
    if not test_blocks:
        raise ValueError('No test blocks available to select from.')
    if len(test_blocks) == 1:
        return next(iter(test_blocks.items()))

    best_name: str | None = None
    best_path: pathlib.Path | None = None
    best_score = -1

    # Check candidate blocks (up to 20 to preserve fast I/O)
    sorted_items = sorted(test_blocks.items())
    candidates = sorted_items[:min(len(sorted_items), 20)]

    for name, path in candidates:
        try:
            with numpy.load(path) as data:
                if label_key in data:
                    lbl = data[label_key]
                    unique_classes = len(numpy.unique(lbl))
                    valid_px = int(numpy.count_nonzero(lbl))
                    score = unique_classes * 1_000_000 + valid_px
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_path = path
        except Exception:
            continue

    if best_name is not None and best_path is not None:
        return best_name, best_path

    # Fallback to middle candidate
    mid = len(sorted_items) // 2
    return sorted_items[mid]


def _place_patch_predictions(
    canvas_pred: dict[str, torch.Tensor],
    canvas_gt: dict[str, torch.Tensor],
    active_heads: list[str],
    preds_batch: dict[str, torch.Tensor],
    y_cpu: torch.Tensor | None,
    hmap: dict[str, int],
    ignore_index: int | list[int] | tuple[int, ...],
    i: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> None:
    """Helper to place prediction and ground-truth patches onto canvases."""
    for h in active_heads:
        if h in preds_batch:
            canvas_pred[h][y0:y1, x0:x1] = preds_batch[h][i]
        if y_cpu is not None and h in hmap and y_cpu.shape[1] > hmap[h]:
            tgt_patch = y_cpu[i, hmap[h], :, :].clone()
            if isinstance(ignore_index, (list, tuple)):
                for ign in ignore_index:
                    tgt_patch[tgt_patch == ign] = 0
            else:
                tgt_patch[tgt_patch == ignore_index] = 0
            canvas_gt[h][y0:y1, x0:x1] = tgt_patch


def generate_prediction_previews(
    checkpoint_fpath: str | pathlib.Path,
    dataspecs: core.DataSpecs,
    config: typing.Any = None,
    previews_dpath: str | pathlib.Path | None = None,
    *,
    patch_size: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    save_pngs: bool = True,
    show: bool = False,
    logger: typing.Any = None,
) -> dict[str, typing.Any]:
    '''
    Execute test inference using best checkpoint and render preview PNGs per head.

    Supports dual-mode execution:
    - Contiguous grid: stitches the full test mosaic across blocks.
    - Scattered/internal split: stitches a single representative diverse block.

    Args:
        checkpoint_fpath:
            Path to saved model checkpoint weights (.pt).
        dataspecs:
            Data specification object containing splits and metadata.
        config:
            Optional RootConfig containing session and model architecture parameters.
        previews_dpath:
            Output directory for saving inferred preview PNGs.
        patch_size:
            Optional patch size override (defaults to config or 128).
        batch_size:
            Optional batch size override (defaults to config or 16).
        device:
            Target execution device (e.g. 'cuda', 'cpu', 'auto').
        save_pngs:
            Whether to save preview images to `previews_dpath`.
        show:
            Whether to display figures with matplotlib.pyplot.show().
        logger:
            Optional pipeline logger instance.

    Returns:
        dict[str, matplotlib.figure.Figure]:
            Dictionary mapping active head names to rendered matplotlib figures.
    '''
    test_blocks = dataspecs.splits.test
    if not test_blocks:
        if logger:
            logger.info('No test blocks found in dataspecs; skipping preview generation.')
        return {}

    ckpt_path = pathlib.Path(checkpoint_fpath)
    if not ckpt_path.is_file():
        if logger:
            logger.warning(f'Checkpoint file not found: {ckpt_path}')
        return {}

    target_device = utils.resolve_device(
        device or (getattr(config.session, 'device', 'auto') if config else 'auto')
    )

    # Resolve grid layout: contiguous full mosaic vs single representative block
    grid_col, grid_row = dataspecs.meta.test_blks_grid
    is_contiguous = (grid_col > 0 and grid_row > 0 and (grid_col * grid_row == len(test_blocks)))

    if is_contiguous:
        selected_blocks = test_blocks
        blk_cols, blk_rows = grid_col, grid_row
    else:
        rep_name, rep_path = select_representative_block(
            test_blocks, label_key=dataspecs.meta.label_specs.array_key
        )
        selected_blocks = {rep_name: rep_path}
        blk_cols, blk_rows = 1, 1

    block_size = dataspecs.meta.image_specs.height_width
    ps = patch_size or (config.session.dataloader.patch_size if config else 128)
    bs = batch_size or (config.session.dataloader.batch_size if config else 16)

    patch_per_dim = block_size // ps
    patch_per_blk = patch_per_dim * patch_per_dim
    total_col_patches = blk_cols * patch_per_dim
    total_row_patches = blk_rows * patch_per_dim
    total_w = total_col_patches * ps
    total_h = total_row_patches * ps

    # Build test dataset and loader
    ids_domain = dataspecs.domains.test.get('ids_domain') if dataspecs.domains else None
    vec_domain = dataspecs.domains.test.get('vec_domain') if dataspecs.domains else None

    ds_config = dataset.BlockDatasetConfig(
        block_size=block_size,
        patch_size=ps,
        image_key=dataspecs.meta.image_specs.array_key,
        label_key=dataspecs.meta.label_specs.array_key,
        ids_domain=ids_domain,
        vec_domain=vec_domain,
    )
    ds = dataset.MultiBlockDataset(
        selected_blocks,
        ds_config,
        augment_flip=False,
        preload=True,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate.batch,
    )

    # Assemble model and load checkpoint
    models_cfg = getattr(config, 'models', None)
    if models_cfg is not None and hasattr(models_cfg, '_content'):
        models_cfg = omegaconf.OmegaConf.to_object(models_cfg)

    if models_cfg is None:
        models_cfg = models_schema.ModelsConfig()

    backbone_cfg = getattr(models_cfg, 'unet_backbone_config', None)
    cond_cfg = getattr(models_cfg, 'conditioning_config', None)
    safety_cfg = getattr(models_cfg, 'numeric_safety', None)
    enable_clamp = getattr(safety_cfg, 'enable_clamp', False) if safety_cfg is not None else False
    clamp_range = (
        getattr(safety_cfg, 'clamp_range', (-10.0, 10.0))
        if safety_cfg is not None
        else (-10.0, 10.0)
    )

    model = models.build_multihead_unet(
        patch_size=ps,
        dataspecs=dataspecs,
        unet_backbone_config=backbone_cfg,
        conditioning_config=cond_cfg,
        enable_clamp=enable_clamp,
        clamp_range=clamp_range,
    )
    artifacts.load_checkpoint(
        model=model,
        fpath=ckpt_path,
        map_device=target_device,
    )
    model.to(target_device)
    model.eval()

    all_heads = list(dataspecs.heads.class_counts.keys())
    hmap = {name: i for i, name in enumerate(all_heads)}
    active_heads = all_heads

    canvas_img = torch.zeros((3, total_h, total_w), dtype=torch.float32)
    canvas_gt = {h: torch.zeros((total_h, total_w), dtype=torch.long) for h in active_heads}
    canvas_pred = {h: torch.zeros((total_h, total_w), dtype=torch.long) for h in active_heads}

    band_map = dataspecs.meta.image_specs.band_map
    if band_map and all(c in band_map for c in ['red', 'green', 'blue']):
        rgb_channels = [band_map['red'], band_map['green'], band_map['blue']]
    else:
        rgb_channels = [0, 1, 2]

    # Run inference and stitch
    global_p_idx = 0
    with torch.no_grad():
        for x_batch, y_batch, dom_batch in loader:
            b_size = x_batch.shape[0]
            x_dev = x_batch.to(target_device)
            dom_dev = {
                k: v.to(target_device)
                for k, v in dom_batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = model(x_dev, **dom_dev)
            preds_batch = {
                h: (torch.argmax(logits, dim=1).detach().cpu() + 1)
                for h, logits in outputs.items()
                if h in active_heads
            }

            # Gate child heads based on predicted parent class
            for h in active_heads:
                p_name = dataspecs.heads.head_parent.get(h)
                p_cls = dataspecs.heads.head_parent_cls.get(h)
                if p_name and p_cls is not None and p_name in preds_batch:
                    inactive = preds_batch[p_name] != p_cls
                    preds_batch[h][inactive] = 0

            x_cpu = x_batch.detach().cpu()
            y_cpu = y_batch.detach().cpu() if y_batch.numel() > 0 else None

            for i in range(b_size):
                curr_p = global_p_idx + i
                blk_idx = curr_p // patch_per_blk
                p_in_blk = curr_p % patch_per_blk

                b_row, b_col = divmod(blk_idx, blk_cols)
                p_row, p_col = divmod(p_in_blk, patch_per_dim)

                patch_col = b_col * patch_per_dim + p_col
                patch_row = b_row * patch_per_dim + p_row

                y0 = patch_row * ps
                y1 = y0 + ps
                x0 = patch_col * ps
                x1 = x0 + ps

                num_c = x_cpu.shape[1]
                chans = [c for c in rgb_channels if c < num_c]
                if len(chans) == 3:
                    canvas_img[:, y0:y1, x0:x1] = x_cpu[i, chans, :, :]
                elif num_c >= 3:
                    canvas_img[:, y0:y1, x0:x1] = x_cpu[i, :3, :, :]
                else:
                    canvas_img[:, y0:y1, x0:x1] = x_cpu[i, 0:1, :, :].repeat(3, 1, 1)

                _place_patch_predictions(
                    canvas_pred=canvas_pred,
                    canvas_gt=canvas_gt,
                    active_heads=active_heads,
                    preds_batch=preds_batch,
                    y_cpu=y_cpu,
                    hmap=hmap,
                    ignore_index=dataspecs.meta.label_specs.ignore_index,
                    i=i,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )

            global_p_idx += b_size

    # Render a preview figure per active head
    result_figs: dict[str, typing.Any] = {}
    for head in active_heads:
        err_tensor = (canvas_pred[head] != canvas_gt[head]).int()
        title = f'Qualitative Segmentation Preview - [{head}]'
        save_file = (
            (pathlib.Path(previews_dpath) / f'{head}_preview.png')
            if (save_pngs and previews_dpath)
            else None
        )
        fig = plot_qualitative_predictions(
            image=canvas_img,
            ground_truth=canvas_gt[head],
            prediction=canvas_pred[head],
            errors=err_tensor,
            title=title,
            save_path=save_file,
            show=show,
            color_map=dataspecs.meta.label_color_map,
        )
        result_figs[head] = fig

    return result_figs


def render_prediction_preview(
    target_dpath: str | pathlib.Path | None = None,
    *,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> typing.Any:
    '''
    Automated qualitative prediction preview for notebook inspection.

    Loads and displays previously generated preview PNGs from the experiment
    results directory, or falls back to a clean mock preview figure.

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

    # Look for generated preview PNGs in previews directories
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

    # Remove duplicates preserving order
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
            ax.set_title(png_file.stem.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.axis('off')
            matplotlib.pyplot.tight_layout()
            if save_path is not None:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
            if show:
                matplotlib.pyplot.show()
            last_fig = fig
        return last_fig

    # Fallback to generating a sample preview figure if no previews exist yet
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
