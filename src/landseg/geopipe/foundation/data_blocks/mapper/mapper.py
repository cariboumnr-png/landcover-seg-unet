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
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.mapper as mapper
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class MappedRasterWindows:
    '''Container for output raster read windows and meta.'''
    grid_id: str                    # world grid identifier
    tile_shape: tuple[int, int]     # expected window shape (W*H) in px
    image: alias.RasterWindowDict   # indexed read windows
    label: alias.RasterWindowDict   # indexed read windows (can be empty)

# -------------------------------Public Function-------------------------------
def map_rasters(
    world_grid: geo_core.GridLayout,
    image_fpath: str,
    label_fpath: str,
    logger: utils.Logger,
) -> MappedRasterWindows:
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

    # get geometry summary
    geom = mapper.validate_geometry(image_fpath, label_fpath, logger)

    # alignment to the world grid and check CRS match
    grid_crs = world_grid.crs
    data_crs = geom['crs']
    if grid_crs != data_crs:
        e = f'Data CRS [{data_crs}] != world CRS [{grid_crs}]'
        logger.log('ERROR', e)
        raise ValueError(e)

    # focus on tiles inside the intersected extent
    inside = _crop(world_grid, geom)

    # get image and optionally label reading windows
    return MappedRasterWindows(
        image=_get_windows(world_grid, geom['image_transform'], inside),
        label=_get_windows(world_grid, geom['label_transform'], inside),
        grid_id=world_grid.gid,
        tile_shape=world_grid.tile_size
    )

# ------------------------------private  function------------------------------
def _crop(
    world_grid: geo_core.GridLayout,
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
    world_grid: geo_core.GridLayout,
    transform: alias.RasterTransform,
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
