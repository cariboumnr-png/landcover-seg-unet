'''Preview.'''

# standard imports
import os
# third-party imports
import numpy
import PIL.Image
import torch

# -------------------------------Public Function-------------------------------
def export_previews(
    maps: dict[str, dict[tuple[int, int], torch.Tensor]],
    map_grid_shape: tuple[int, int],
    out_dir: str,
    heads: list[str] | None = None,
    palettes: dict[str, numpy.ndarray] | None = None,
) -> dict[str, str]:
    '''
    Build and save PNG previews per head from infer maps.

    Args:
        maps: { blkname: { head: Tensor[H_blk, W_blk] } }
        out_dir: directory to write PNGs into.
        heads: which heads to export; if None, use all heads in maps.
        index_base: 0 if names start with col_0/row_0; set to 1 for
            col_1/row_1.
        palettes: optional per-head palettes { head: [num_classes, 3] }.

    Returns:
        { head: path_to_png }
    '''

    # make output dir if not already
    os.makedirs(out_dir, exist_ok=True)

    # parse grid shape
    cols_total, rows_total = map_grid_shape

    # if heads not provided, use all heads
    if heads is None:
        heads = list(maps.keys())

    # iterate through heads
    results: dict[str, str] = {}
    for head in heads:
        mosaic = _stitch_patches(maps[head], cols_total, rows_total)
        pal = palettes.get(head) if palettes else None
        out_path = f'{out_dir}/preview_{head}.png'
        _save_index_mosaic_png(mosaic, out_path, pal)
        results[head] = out_path

    return results

# ------------------------------private  function------------------------------
# ----- mosaic building

def _stitch_patches(
    placements: dict[tuple[int, int], torch.Tensor],
    cols_total: int,
    rows_total: int,
    *,
    fill_value: int | float = 0,
) -> torch.Tensor:
    '''
    Merge {(col, row): patch[Hp, Wp]} into a full [H_total, W_total] tensor.

    cols_total, rows_total are the total number of patches horizontally
    and vertically (precomputed upstream).

    Returns:
        Canvas of shape [H_total, W_total].
    '''
    assert placements, 'placements is empty'

    # Take a sample patch to get shape/device/dtype
    any_patch = next(iter(placements.values()))
    hp, wp = any_patch.shape
    device = any_patch.device
    dtype = any_patch.dtype

    # init a canvas
    canvas = torch.full(
        (rows_total * hp, cols_total * wp),
        fill_value=fill_value,
        dtype=dtype,
        device=device,
    )

    # place each patch and return
    for (col, row), patch in placements.items():
        y = row * hp
        x = col * wp
        canvas[y: y + hp, x: x + wp] = patch
    return canvas

# ----- png saving
def _save_index_mosaic_png(
    mosaic: torch.Tensor,
    out_path: str,
    palette: numpy.ndarray | None = None
) -> None:
    '''
    Save a class-index mosaic tensor as a color PNG.

    Notes:
        - Detaches and moves to CPU exactly once here.
        - Expects mosaic to contain integer class indices.
    '''

    arr = mosaic.detach().to('cpu').numpy()
    if palette is None:
        max_cls = int(numpy.max(arr)) if arr.size > 0 else 0
        palette = _default_palette(max_cls + 1)

    rgb = _colorize_indices(arr, palette)
    PIL.Image.fromarray(rgb, mode='RGB').save(out_path)

def _default_palette(
    num_classes: int,
    seed: int = 123
) -> numpy.ndarray:
    '''
    Generate a deterministic random RGB palette [num_classes, 3] uint8.
    '''

    num_classes = max(1, int(num_classes))
    rng = numpy.random.default_rng(seed)
    pal = rng.integers(0, 256, size=(num_classes, 3), dtype=numpy.uint8)
    # Improve contrast for the first few classes (optional tweak)
    if num_classes >= 3:
        pal[0] = numpy.array([0, 0, 0], dtype=numpy.uint8)   # backg -> black
        pal[1] = numpy.array([255, 0, 0], dtype=numpy.uint8) # class 1 -> red
        pal[2] = numpy.array([0, 255, 0], dtype=numpy.uint8) # class 2 -> green
    return pal

def _colorize_indices(
    index_map: numpy.ndarray,
    palette: numpy.ndarray
) -> numpy.ndarray:
    '''
    Map a 2D class index array [H, W] to RGB [H, W, 3] using a palette.

    Args:
        index_map: 2D integer array.
        palette: [num_classes, 3] uint8.

    Returns:
        RGB array [H, W, 3] uint8.
    '''

    if index_map.ndim != 2:
        raise ValueError('index_map must be 2D [H, W]')
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError('palette must be shaped [num_classes, 3]')

    idx = numpy.asarray(index_map)
    if not numpy.issubdtype(idx.dtype, numpy.integer):
        idx = idx.astype(numpy.int32)

    idx = numpy.clip(idx, 0, palette.shape[0] - 1)
    rgb = palette[idx]
    return rgb.astype(numpy.uint8)
