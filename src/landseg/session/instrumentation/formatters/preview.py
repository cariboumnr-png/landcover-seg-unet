# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

'''Preview.'''

# third-party imports
import numpy
import PIL.Image
import torch

# -------------------------------Public Function-------------------------------
def stitch_patches(
    placements: dict[tuple[int, int], torch.Tensor],
    *,
    grid_shape: tuple[int, int] | None = None,
    palette: numpy.ndarray | dict[int, list[int]] | None = None
) -> numpy.ndarray:
    '''
    Merge {(col, row): patch[Hp, Wp]} into a full [H_total, W_total] tensor.

    cols_total, rows_total are the total number of patches horizontally
    and vertically (precomputed upstream).

    Returns:
        Canvas of shape [H_total, W_total].
    '''
    assert placements, 'placements is empty'

    # Take a sample patch to get shape
    any_patch = next(iter(placements.values()))
    if any_patch.dim() == 2:
        hp, wp = any_patch.shape
    elif any_patch.dim() == 3:
        _, hp, wp = any_patch.shape
    else:
        raise ValueError(f'Unsupported tensor shape: {any_patch.shape}')

    # if total cols and rows are provided, infer from the placement dict
    if grid_shape is None:
        cols_total = max(col for col, _ in placements) + 1
        rows_total = max(row for _, row in placements) + 1
    else:
        cols_total, rows_total = grid_shape

    # init a canvas
    canvas = torch.full(
        (rows_total * hp, cols_total * wp),
        fill_value=0,
        dtype=any_patch.dtype,
        device=any_patch.device,
    )

    # place each patch according to the grid coordinates
    for (col, row), patch in placements.items():
        y = row * hp
        x = col * wp
        canvas[y: y + hp, x: x + wp] = patch

    #
    array = canvas.detach().to('cpu').numpy()
    if palette is None:
        max_cls = int(numpy.max(array)) if array.size > 0 else 0
        palette = _default_palette(max_cls + 1)
    elif isinstance(palette, dict):
        palette = _palette_from_dict(palette)
    else:
        pass
    return _colorize_indices(array, palette)

def _default_palette(
    num_classes: int,
    *,
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
    Map a 2D class index array [H, W] to RGB [3, H, W] using a palette.

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
    rgb = palette[idx] # HWC
    rgb = numpy.transpose(rgb, (2, 0, 1)) # CHW
    return rgb.astype(numpy.uint8)

def _palette_from_dict(
    color_map: dict[int, list[int]],
    default_color: tuple[int, int, int] = (0, 0, 0)
) -> numpy.ndarray:
    '''
    Convert a mapping of class index -> RGB color into a palette array.
    '''

    if not color_map:
        raise ValueError('color_map cannot be empty')

    # key type guard
    color_map = {int(k): v for k, v in color_map.items()}

    max_index = max(color_map.keys())

    palette = numpy.full(
        (max_index + 1, 3),
        default_color,
        dtype=numpy.uint8
    )

    for class_index, rgb in color_map.items():

        if len(rgb) != 3:
            raise ValueError(
                f'RGB value for class {class_index} must contain 3 elements'
            )

        palette[class_index] = numpy.asarray(rgb, dtype=numpy.uint8)

    return palette

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
