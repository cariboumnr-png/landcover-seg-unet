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

'''Preview utilities'''

# third-party imports
import numpy
import torch

# -------------------------------Public Function-------------------------------
def stitch_patches(
    placements: dict[tuple[int, int], torch.Tensor],
    *,
    grid_shape: tuple[int, int] | None = None,
    palette: torch.Tensor | numpy.ndarray | dict[int, list[int]] | None = None
) -> torch.Tensor:
    '''
    Merge {(col, row): patch[Hp, Wp]} into a full RGB tensor.

    Returns:
        RGB tensor of shape [3, H_total, W_total] uint8.
    '''

    assert placements, 'placements is empty'
    # sample patch
    any_patch = next(iter(placements.values()))
    if any_patch.dim() == 2:
        hp, wp = any_patch.shape
    elif any_patch.dim() == 3:
        _, hp, wp = any_patch.shape
    else:
        raise ValueError(f'Unsupported tensor shape: {any_patch.shape}')

    # infer grid size
    if grid_shape is None:
        cols_total = max(col for col, _ in placements) + 1
        rows_total = max(row for _, row in placements) + 1
    else:
        cols_total, rows_total = grid_shape

    # allocate mosaic index canvas
    canvas = torch.full(
        (rows_total * hp, cols_total * wp),
        fill_value=0,
        dtype=any_patch.dtype,
        device=any_patch.device,
    )

    # stitch patches
    for (col, row), patch in placements.items():

        if patch.dim() == 3:
            patch = patch.squeeze(0)

        y = row * hp
        x = col * wp

        canvas[y:y + hp, x:x + wp] = patch

    # palette handling
    if palette is None:
        max_cls = int(canvas.max().item()) if canvas.numel() > 0 else 0
        palette = _default_palette(max_cls + 1, device=canvas.device)

    elif isinstance(palette, dict):
        palette = _palette_from_dict(
            palette,
            device=canvas.device
        )

    elif isinstance(palette, numpy.ndarray):
        palette = torch.as_tensor(
            palette,
            dtype=torch.uint8,
            device=canvas.device
        )

    return _colorize_indices(canvas, palette)

def _default_palette(
    num_classes: int,
    *,
    seed: int = 123,
    device: torch.device | None = None
) -> torch.Tensor:
    '''Generate deterministic RGB palette [N, 3] uint8 tensor.'''

    num_classes = max(1, int(num_classes))

    rng = numpy.random.default_rng(seed)

    pal = rng.integers(
        0,
        256,
        size=(num_classes, 3),
        dtype=numpy.uint8
    )

    if num_classes >= 3:
        pal[0] = [0, 0, 0]
        pal[1] = [255, 0, 0]
        pal[2] = [0, 255, 0]

    return torch.as_tensor(
        pal,
        dtype=torch.uint8,
        device=device
    )

def _colorize_indices(
    index_map: torch.Tensor,
    palette: torch.Tensor
) -> torch.Tensor:
    '''Map class indices [H, W] -> RGB tensor [3, H, W].'''

    if index_map.dim() != 2:
        raise ValueError('index_map must be 2D [H, W]')

    if palette.dim() != 2 or palette.shape[1] != 3:
        raise ValueError('palette must be shaped [N, 3]')

    idx = index_map.to(torch.long)

    idx = torch.clamp(
        idx,
        0,
        palette.shape[0] - 1
    )

    rgb = palette[idx]          # [H, W, 3]
    rgb = rgb.permute(2, 0, 1) # [3, H, W]

    return rgb.contiguous()


def _palette_from_dict(
    color_map: dict[int, list[int]],
    default_color: tuple[int, int, int] = (0, 0, 0),
    *,
    device: torch.device | None = None
) -> torch.Tensor:
    '''Convert class->RGB dict into palette tensor [N, 3].'''

    if not color_map:
        raise ValueError('color_map cannot be empty')

    color_map = {int(k): v for k, v in color_map.items()}

    max_index = max(color_map.keys())

    palette = torch.full(
        (max_index + 1, 3),
        fill_value=0,
        dtype=torch.uint8,
        device=device
    )

    palette[:] = torch.tensor(
        default_color,
        dtype=torch.uint8,
        device=device
    )

    for class_index, rgb in color_map.items():

        if len(rgb) != 3:
            raise ValueError(
                f'RGB value for class {class_index} must contain 3 elements'
            )

        palette[class_index] = torch.tensor(
            rgb,
            dtype=torch.uint8,
            device=device
        )

    return palette
