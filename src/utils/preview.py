'''Preview.'''

# standard imports
import os
import re
# third-party imports
import numpy
import PIL.Image
import torch

# -------------------------------Public Function-------------------------------
def export_previews(
    maps: dict[str, dict[str, torch.Tensor]],
    out_dir: str,
    heads: list[str] | None = None,
    palettes: dict[str, numpy.ndarray] | None = None,
    index_base: int = 0,
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

    # if heads not provided, use all heads
    if heads is None:
        heads = sorted(_discover_heads(maps))

    results: dict[str, str] = {}
    for head in heads:
        mosaic = _build_mosaic_per_head(maps, head, index_base)
        pal = palettes.get(head) if palettes else None
        out_path = f'{out_dir}/preview_{head}.png'
        _save_index_mosaic_png(mosaic, out_path, pal)
        results[head] = out_path

    return results

# ------------------------------private  function------------------------------
# ----- parse heads
def _discover_heads(maps: dict[str, dict[str, torch.Tensor]]) -> set[str]:
    '''
    Return the set of head names present in the infer maps.
    '''

    heads: set[str] = set()
    for per_blk in maps.values():
        heads.update(per_blk.keys())
    return heads

# ----- mosaic building
def _build_mosaic_per_head(
    maps: dict[str, dict[str, torch.Tensor]],
    head: str,
    index_base: int=0
) -> torch.Tensor:
    '''
    Stitch a full-scene mosaic for a specific head from infer maps.

    Args:
        maps: { blkname: { head: Tensor[H_blk, W_blk] } }
        head: head name to assemble.
        index_base: 0 if names start at col_0/row_0, 1 if names start at
            col_1/row_1.

    Returns:
        Tensor[H_full, W_full] with class indices (same dtype/device as
            inputs).
    '''

    # extract from given head
    items: list[tuple[str, torch.Tensor]] = []
    for blkname, per_head in maps.items():
        if head in per_head:
            items.append((blkname, per_head[head]))
    if not items:
        raise ValueError(f'No blocks found for head="{head}"')

    # infer block size and device/dtype from the first block
    _, sample = items[0]
    h_blk, w_blk = int(sample.shape[0]), int(sample.shape[1])
    device, dtype = sample.device, sample.dtype

    # sanity: ensure all blocks same shape (optional but helpful)
    for _, block in items:
        if block.shape != sample.shape:
            raise ValueError(
                'Inconsistent block shapes for head '
                f'"{head}": expected {tuple(sample.shape)}, '
                f'got {tuple(block.shape)}'
            )

    # parse coords and compute grid size
    coords = [_parse_col_row(n) for (n, _) in items]
    max_col = max(c for c, _ in coords)
    max_row = max(r for _, r in coords)
    grid_w = max_col - index_base + 1
    grid_h = max_row - index_base + 1
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError(
            f'invalid grid size from names: grid=({grid_h}, {grid_w}), '
            f'index_base={index_base}'
        )

    # int with background (0); avoids uninitialized garbage.
    mosaic = torch.zeros(
        (grid_h * h_blk, grid_w * w_blk),
        dtype=dtype,
        device=device
    )

    # Place each block into the mosaic and return
    for blkname, block in items:
        col, row = _parse_col_row(blkname)
        r0 = (row - index_base) * h_blk
        c0 = (col - index_base) * w_blk
        mosaic[r0: r0 + h_blk, c0: c0 + w_blk] = block
    return mosaic

def _parse_col_row(name: str) -> tuple[int, int]:
    '''
    Extract (col, row) from a block name like 'col_03_row_05'.
    Supports minor variations with dashes and case.
    '''

    pattern = re.compile(r'.*col[_-]?(\d+).*row[_-]?(\d+).*', re.IGNORECASE)
    m = pattern.match(name)
    if not m:
        raise ValueError(f'cannot parse (col,row) from block name: {name}')
    col, row = int(m.group(1)), int(m.group(2))
    return int(col / 256), int(row / 256)

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
