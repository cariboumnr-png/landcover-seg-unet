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

'''
Raster-to-grid mapping utilities that generate indexed read windows for
images and labels, and persist them as artifacts for training/inference.

Public APIs:
    - map_rasters: Map input rasters onto the world grid and serialize
      window indices for fit/test datasets.'
'''

# standard imports
import copy
import dataclasses
import os
# third-party imports
import rasterio
# local imports
import landseg.alias as alias
import landseg.core as core
import landseg.dataset as dataset
import landseg.dataset.mapper as mapper
import landseg.utils as utils

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class DataWindows:
    '''Container for input read windows and expected window shape.'''
    image_windows: alias.RasterWindowDict   # indexed read windows
    label_windows: alias.RasterWindowDict   # indexed read windows can be empty
    expected_shape: tuple[int, int]         # expected window shape (W*H) in px

# -------------------------------Public Function-------------------------------
def map_rasters(
    world_grid: core.GridLayoutLike,
    config: dataset.IOConfig,
    logger: utils.Logger,
    *,
    remap: bool = False
) -> None:
    '''
    Map input rasters to the world grid and serialize read windows.

    Args:
        world_grid: Target grid layout definition.
        config: I/O config with input paths and output artifact targets.
        logger: Logger for progress and diagnostics.
        remap: If True, force recompute mappings.

    Raises:
        ValueError: If the rasters' CRS does not match the world grid CRS.
    '''

    # paths to artifacts from fit input
    fit_img = config['fit_input_img']
    fit_lbl = config['fit_input_lbl']
    fit_windows_fpath = config['fit_windows']
    # map fit data
    if not os.path.exists(fit_windows_fpath) or remap:
        _map(world_grid, fit_img, fit_lbl, fit_windows_fpath, logger)

    # paths to artifacts from test input if provided
    test_img = config['test_input_img']
    test_lbl = config['test_input_lbl']
    test_windows_fpath = config['test_windows']
    # map test data if provided
    if test_img and (not os.path.exists(test_windows_fpath) or remap):
        _map(world_grid, test_img, test_lbl, test_windows_fpath, logger)

def _map(
    world_grid: core.GridLayoutLike,
    image_fpath: str,
    label_fpath: str | None,
    windows_fpath: str,
    logger: utils.Logger
) -> None:
    '''Create and persist windows for a raster pair (image & label).'''

    # get geometry summary
    geom = mapper.validate_geometry(image_fpath, label_fpath, logger)

    # alignment to the world grid and check CRS match
    grid_crs = world_grid.crs
    data_crs = geom['crs']
    if grid_crs != data_crs:
        raise ValueError(f'Data CRS [{data_crs}] != world CRS [{grid_crs}]')

    # focus on tiles inside the intersected extent
    inside = _crop(world_grid, geom)

    # get image and optionally label reading windows
    img_windows = _get_windows(world_grid, geom['image_transform'], inside)
    lbl_windows = _get_windows(world_grid, geom['label_transform'], inside)

    # pickle
    data = DataWindows(img_windows, lbl_windows, world_grid.tile_size)
    utils.write_pickle(windows_fpath, data)
    utils.hash_artifacts(windows_fpath)

# ------------------------------private  function------------------------------
def _crop(
    world_grid: core.GridLayoutLike,
    geom_summary: mapper.GeometrySummary,
) -> list[tuple[int, int]]:
    '''Return grid tile indices that intersect the raster extent.'''

    # prep return list
    inside: list[tuple[int, int]] = []
    # get world origin
    gx, gy = world_grid.origin
    # get pixel size and bbox
    px, py = geom_summary['pixel_size'] # in CRS units per pixel
    l, b, r, t = geom_summary['inter_bbox'] # in CRS units
    # get tile size to CRS units
    tile_x = world_grid.tile_size[1] * px
    tile_y = world_grid.tile_size[0] * py
    # iterate through world grid
    for (x, y) in world_grid.keys(): # keys (x, y) are in pixels
        # unify comparisons to be between CRS coordinates
        if (
            gx + x * px < r and             # right side of tile < R
            gy - y * py > b and             # bottom side of tile > B
            gx + x * px > l - tile_x and    # left side of tile > L
            gy - y * py < t + tile_y        # top side of tile < T
        ):
            inside.append((x, y))
    return inside

def _get_windows(
    world_grid: core.GridLayoutLike,
    transform: rasterio.Affine | None,
    inside_idx: list[tuple[int, int]]
) -> alias.RasterWindowDict:
    '''Return window dict for tiles inside the target area .'''

    # get raster reading windows - empty when transform is None
    windows: alias.RasterWindowDict = {}
    if transform is not None:
        # set grid offset for label
        _grid = copy.deepcopy(world_grid)
        _grid.offset_from(transform)
        # get windows only inside
        for idx, window in _grid.items():
            if idx in inside_idx:
                windows[idx] = window
    # return
    return windows
