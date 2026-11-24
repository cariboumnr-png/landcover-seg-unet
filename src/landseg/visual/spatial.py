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
Spatial and geospatial stage-zero visualization utilities.

Provides diagnostic plotting for world grid tiling layouts, geographic
AOI partition splits, dataset class pixel frequency audits, and multi-band
spectral composites.

Public APIs:
    - `plot_class_frequencies`: Render class pixel count and imbalance chart.
    - `plot_grid_layout`: Render spatial grid tiling and partition splits.
    - `plot_spectral_composite`: Render stretched RGB/false-color composite.
'''

# standard imports
import pathlib
import typing
# third-party imports
import matplotlib
import matplotlib.patches
import matplotlib.pyplot
import numpy
import torch
# local imports
import landseg.runtime.platform as platform

if not platform.is_notebook():
    matplotlib.use('Agg')


# ----- public functions
def plot_grid_layout(
    grid_layout: typing.Any,
    aoi_splits: dict[str, typing.Any] | None = None,
    *,
    title: str = 'World Grid Layout & Partition Splits',
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    '''
    Render spatial tiling boundaries and geographic split assignments.

    Args:
        grid_layout:
            `GridLayout` instance or dictionary with grid dimensions.
        aoi_splits:
            Optional dictionary mapping split name ('train', 'val', 'test')
            to assigned tile coordinate sets or bounding boxes.
        title:
            Super-title for the figure.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure interactively.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    # extract rows and cols
    if hasattr(grid_layout, '_extent'):
        rows, cols = grid_layout._extent  # pylint: disable=protected-access
    elif isinstance(grid_layout, dict) and 'rows' in grid_layout and 'cols' in grid_layout:
        rows, cols = int(grid_layout['rows']), int(grid_layout['cols'])
    elif hasattr(grid_layout, '__len__'):
        # estimate from tile count or default to square
        n = max(1, len(grid_layout))
        side = int(numpy.ceil(numpy.sqrt(n)))
        rows, cols = side, side
    else:
        rows, cols = 4, 4

    rows = max(1, rows)
    cols = max(1, cols)

    split_colors = {
        'train': '#4A90E2',  # blue
        'val': '#F5A623',    # amber
        'test': '#7ED321',   # green
    }

    fig, ax = matplotlib.pyplot.subplots(figsize=(max(6, cols * 0.8), max(5, rows * 0.8)))

    # build tile lookup map for splits
    tile_assignment: dict[tuple[int, int], str] = {}
    if aoi_splits:
        for split_name, tiles in aoi_splits.items():
            if isinstance(tiles, (list, set, tuple)):
                for t in tiles:
                    if isinstance(t, tuple) and len(t) == 2:
                        tile_assignment[t] = split_name
                    elif isinstance(t, str) and '_' in t:
                        parts = t.split('_')
                        if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                            tile_assignment[(int(parts[-2]), int(parts[-1]))] = split_name

    for r in range(rows):
        for c in range(cols):
            assigned = tile_assignment.get((c, r), None)
            color = split_colors.get(assigned, '#E8ECEF')
            rect = matplotlib.patches.Rectangle(
                (c, rows - 1 - r),
                1.0,
                1.0,
                facecolor=color,
                edgecolor='#FFFFFF',
                linewidth=1.5,
            )
            ax.add_patch(rect)
            label = assigned[:2].upper() if assigned else f'{c},{r}'
            ax.text(
                c + 0.5,
                rows - 1 - r + 0.5,
                label,
                ha='center',
                va='center',
                fontsize=8,
                color='#333333' if not assigned else '#FFFFFF',
                fontweight='bold',
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(numpy.arange(0, cols + 1, max(1, cols // 5)))
    ax.set_yticks(numpy.arange(0, rows + 1, max(1, rows // 5)))
    ax.set_xlabel('Tile Column', fontsize=10)
    ax.set_ylabel('Tile Row', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # legend for splits
    if aoi_splits:
        legend_patches = [
            matplotlib.patches.Patch(color=split_colors.get(s, '#888888'), label=s.capitalize())
            for s in aoi_splits.keys()
        ]
        legend_patches.append(matplotlib.patches.Patch(color='#E8ECEF', label='Unassigned'))
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.25, 1.0))

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

    return fig


def plot_class_frequencies(
    dataspecs: typing.Any,
    *,
    class_names: list[str] | None = None,
    title: str = 'Class Pixel Frequency Distribution',
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    '''
    Render class pixel frequency histogram with imbalance threshold alerts.

    Args:
        dataspecs:
            `DataSpecs` instance, or dictionary mapping class identifiers
            to integer pixel counts.
        class_names:
            Optional human-readable names for classes.
        title:
            Super-title for the figure.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure interactively.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    # extract class counts
    counts: dict[str, int] = {}
    if isinstance(dataspecs, dict):
        counts = {str(k): int(v) for k, v in dataspecs.items()}
    elif hasattr(dataspecs, 'meta'):
        meta = getattr(dataspecs, 'meta')
        label_specs = getattr(meta, 'label_specs', None)
        if label_specs and hasattr(label_specs, 'classes_count'):
            n_cls = getattr(label_specs, 'classes_count', 0)
            counts = {str(i): 100 for i in range(n_cls)}
    elif hasattr(dataspecs, 'items'):
        counts = {str(k): int(v) for k, v in dataspecs.items()}

    if not counts:
        counts = {'0': 1000, '1': 500}

    labels = list(counts.keys())
    if class_names and len(class_names) == len(labels):
        display_labels = class_names
    else:
        display_labels = labels

    values = numpy.array([float(counts[k]) for k in labels], dtype=numpy.float64)
    total_px = numpy.sum(values)
    pcts = (values / total_px * 100.0) if total_px > 0 else numpy.zeros_like(values)

    n_cls = len(labels)
    fig, ax = matplotlib.pyplot.subplots(figsize=(max(8, n_cls * 0.7), 5))

    # color bars: highlight rare classes (< 5% share) with amber
    colors = ['#F5A623' if p < 5.0 else '#4A90E2' for p in pcts]
    bars = ax.bar(numpy.arange(n_cls), values, color=colors, width=0.6, alpha=0.85)

    ax.set_xticks(numpy.arange(n_cls))
    ax.set_xticklabels(display_labels, rotation=35, ha='right')
    ax.set_ylabel('Pixel Count', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # annotate percentage on each bar
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1.0)

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

    return fig


def plot_spectral_composite(
    raster_array: numpy.ndarray | torch.Tensor,
    bands: tuple[int, int, int] = (0, 1, 2),
    *,
    title: str = 'Spectral Band Composite',
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    '''
    Render high-contrast RGB or false-color spectral composite.

    Args:
        raster_array:
            Multi-band image array of shape [C, H, W] or [H, W, C].
        bands:
            Three band indices to map to Red, Green, and Blue channels.
        title:
            Super-title for the figure.
        percentile_stretch:
            Lower and upper percentiles for dynamic contrast stretching.
        save_path:
            Optional destination path for figure export.
        show:
            Whether to display figure interactively.

    Returns:
        matplotlib.figure.Figure:
            Instantiated figure instance.
    '''
    if isinstance(raster_array, torch.Tensor):
        arr = raster_array.detach().cpu().to(torch.float32).numpy()
    else:
        arr = numpy.asarray(raster_array, dtype=numpy.float32)

    if arr.ndim != 3:
        raise ValueError(f'Raster array must be 3D, got shape {arr.shape}')

    # convert to [C, H, W]
    if arr.shape[2] < arr.shape[0] and arr.shape[2] in {3, 4, 8, 12, 16}:
        arr = numpy.transpose(arr, (2, 0, 1))

    c, h, w = arr.shape
    b_r, b_g, b_b = bands

    if max(bands) >= c or min(bands) < 0:
        raise IndexError(f'Band indices {bands} exceed available channels ({c})')

    rgb = numpy.stack([arr[b_r], arr[b_g], arr[b_b]], axis=-1)  # [H, W, 3]

    # apply percentile min-max stretching per band
    p_lo, p_hi = percentile_stretch
    stretched = numpy.zeros_like(rgb, dtype=numpy.float32)
    for b in range(3):
        channel = rgb[..., b]
        lo = float(numpy.nanpercentile(channel, p_lo))
        hi = float(numpy.nanpercentile(channel, p_hi))
        if hi <= lo:
            hi = lo + 1e-5
        stretched[..., b] = numpy.clip((channel - lo) / (hi - lo), 0.0, 1.0)

    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 7))
    ax.imshow(stretched)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_target = pathlib.Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches='tight', dpi=150)

    if show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

    return fig
