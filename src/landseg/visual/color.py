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
Color mapping and palette utilities for segmentation visualization.

Public APIs:
    - `DEFAULT_ERROR_PALETTE`: Standard orange/gray error highlight palette.
    - `DEFAULT_MATCH_PALETTE`: Standard green/gray match highlight palette.
    - `FORESTRY_FRI_PALETTE`: Standard Ontario FRI tree species palette.
    - `NLCD_16_PALETTE`: Standard 16-class National Land Cover Database palette.
    - `colorize`: Map 2D class integer masks to 3D RGB tensors.
    - `colorize_continuous`: Map 2D float tensors to 3D RGB colormaps.
'''

# third-party imports
import matplotlib
import numpy
import torch


# ----- public constants
DEFAULT_ERROR_PALETTE: dict[str, list[int]] = {
    '0': [255, 140, 0],  # orange error
    '1': [40, 40, 40],   # dark gray match
}

DEFAULT_MATCH_PALETTE: dict[str, list[int]] = {
    '0': [40, 40, 40],   # dark gray non-match
    '1': [46, 139, 87],  # sea green match
}

NLCD_16_PALETTE: dict[str, list[int]] = {
    '0': [0, 0, 0],         # Background / Unclassified
    '1': [70, 107, 159],    # Open Water
    '2': [209, 222, 248],   # Perennial Ice/Snow
    '3': [222, 197, 197],   # Developed, Open Space
    '4': [217, 146, 130],   # Developed, Low Intensity
    '5': [235, 0, 0],       # Developed, Medium Intensity
    '6': [171, 0, 0],       # Developed, High Intensity
    '7': [179, 172, 159],   # Barren Land
    '8': [104, 171, 95],    # Deciduous Forest
    '9': [28, 99, 48],      # Evergreen Forest
    '10': [181, 201, 143],  # Mixed Forest
    '11': [204, 186, 125],  # Shrub/Scrub
    '12': [227, 227, 194],  # Grassland/Herbaceous
    '13': [220, 217, 57],   # Pasture/Hay
    '14': [171, 112, 40],   # Cultivated Crops
    '15': [184, 217, 235],  # Woody Wetlands
    '16': [108, 169, 199],  # Emergent Herbaceous Wetlands
}

FORESTRY_FRI_PALETTE: dict[str, list[int]] = {
    '0': [0, 0, 0],         # Background
    '1': [34, 139, 34],     # SB - Black Spruce (ForestGreen)
    '2': [46, 139, 87],     # SW - White Spruce (SeaGreen)
    '3': [0, 100, 0],       # PJ - Jack Pine (DarkGreen)
    '4': [107, 142, 35],    # PR - Red Pine (OliveDrab)
    '5': [85, 107, 47],     # PW - White Pine (DarkOliveGreen)
    '6': [144, 238, 144],   # PO - Poplar / Aspen (LightGreen)
    '7': [255, 215, 0],     # BW - White Birch (Gold)
    '8': [218, 165, 32],    # MH - Sugar Maple (GoldenRod)
    '9': [205, 133, 63],    # OR - Red Oak (Peru)
    '10': [64, 224, 208],   # B - Balsam Fir (Turquoise)
    '11': [0, 128, 128],    # CE - Cedar (Teal)
    '12': [70, 130, 180],   # LA - Larch / Tamarack (SteelBlue)
}


# ----- public functions
def colorize(
    canvas: torch.Tensor,
    *,
    palette: torch.Tensor | numpy.ndarray | dict[str, list[int]] | None = None,
) -> torch.Tensor:
    '''Map class indices [H, W] -> RGB tensor [3, H, W].'''
    # palette handling
    if not bool(palette):  # None | empty array, dict, tensor
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

    if canvas.dim() != 2:
        raise ValueError('index_map must be 2D [H, W]')

    if palette.dim() != 2 or palette.shape[1] != 3:
        raise ValueError('palette must be shaped [N, 3]')

    idx = canvas.to(torch.long)
    idx = torch.clamp(idx, 0, palette.shape[0] - 1)

    rgb = palette[idx]          # [H, W, 3]
    rgb = rgb.permute(2, 0, 1)  # [3, H, W]

    return rgb.contiguous()


def colorize_continuous(
    canvas: torch.Tensor,
    *,
    cmap: str = 'viridis',
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    '''
    Map continuous 2D float tensor [H, W] -> [3, H, W] uint8 RGB tensor.

    Useful for continuous prediction heads such as canopy height,
    biomass estimation, or crown density.

    Args:
        canvas:
            Input 2D tensor [H, W].
        cmap:
            Matplotlib colormap identifier (e.g. 'viridis', 'turbo', 'RdYlGn').
        vmin:
            Optional lower normalization boundary.
        vmax:
            Optional upper normalization boundary.

    Returns:
        torch.Tensor:
            Colorized RGB tensor [3, H, W] of dtype uint8.
    '''
    if canvas.dim() != 2:
        raise ValueError('canvas must be 2D [H, W]')

    arr = canvas.detach().cpu().to(torch.float32).numpy()

    if arr.size > 0:
        arr_min = float(numpy.nanmin(arr))
        arr_max = float(numpy.nanmax(arr))
    else:
        arr_min = 0.0
        arr_max = 1.0

    lo = float(vmin) if vmin is not None else arr_min
    hi = float(vmax) if vmax is not None else arr_max

    if hi <= lo:
        hi = lo + 1e-6

    norm = numpy.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    colormap = matplotlib.colormaps.get_cmap(cmap)
    rgba = colormap(norm)  # [H, W, 4] float in [0, 1]
    rgb = (rgba[..., :3] * 255.0).astype(numpy.uint8)  # [H, W, 3]

    res = torch.as_tensor(rgb, device=canvas.device).permute(2, 0, 1)  # [3, H, W]
    return res.contiguous()


# ----- private helpers
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


def _palette_from_dict(
    palette_dict: dict[str, list[int]],
    *,
    device: torch.device | None = None
) -> torch.Tensor:
    '''Map class dictionary to [N, 3] RGB palette tensor.'''
    assert palette_dict, 'Palette dict must not be empty'

    max_idx = max(int(k) for k in palette_dict.keys())
    pal = numpy.zeros((max_idx + 1, 3), dtype=numpy.uint8)

    for k, rgb in palette_dict.items():
        pal[int(k)] = rgb

    return torch.as_tensor(pal, device=device)
