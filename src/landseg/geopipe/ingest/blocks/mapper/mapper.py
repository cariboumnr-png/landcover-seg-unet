# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Raster-to-grid mapping utilities for windowed raster reading.

This module maps input raster datasets onto a standardized world grid
layout, generating windowed read boundaries for image and label bands
used during block construction and model training workflows.

Public APIs:
    - MappedRasterWindows: Dataclass container for read windows.
    - map_rasters: Maps input rasters to grid and builds windows.
'''

# standard imports
import copy
import dataclasses
import pathlib
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.rasters as rasters


# ----- public dataclasses
@dataclasses.dataclass
class MappedRasterWindows:
    '''Container for output raster read windows and metadata.'''
    grid_id: str                        # world grid identifier
    image: geo_core.RasterWindowDict   # indexed read windows
    label: geo_core.RasterWindowDict   # indexed read windows (can be empty)
    tile_shape: tuple[int, int]         # expected window shape (W*H) in px
    tile_overlap: tuple[int, int]


# ----- public functions
def map_rasters(
    world_grid: geo_core.GridLayout,
    image_fpath: pathlib.Path,
    label_fpath: pathlib.Path | None,
) -> MappedRasterWindows:
    '''
    Map input rasters to the world grid and build read windows.

    Aligns input image and optional label rasters with a target world
    grid, checks CRS compatibility, identifies grid cells overlapping
    the intersected raster extent, and computes corresponding raster
    read windows.

    Args:
        world_grid:
            Target spatial grid layout definition.
        image_fpath:
            File path to source image raster.
        label_fpath:
            Optional file path to source label raster, or `None`.

    Returns:
        MappedRasterWindows:
            Container storing mapped image and label window
            dictionaries and grid metadata.

    Raises:
        ValueError:
            If image is missing, CRSs disagree, pixel sizes differ,
            spatial extents do not overlap, or raster CRS does not
            match the target world grid CRS.
    '''
    with rasters.read_rasters(image_fpath, label_fpath) as (img, lbl):
        # make sure at least image is present
        if img is None:
            raise ValueError('A valid image raster is required')
        if img.crs is None:
            raise ValueError('A valid image raster with CRS is required')
        if lbl is not None and lbl.crs is None:
            raise ValueError('A valid label raster with CRS is required')

        # check crs match and alignment with world grid
        data_crs = rasters.check_raster_crs(img.crs, lbl.crs if lbl else None)
        if not rasters.is_same_crs(world_grid.crs, data_crs):
            raise ValueError(
                f'Data CRS [{data_crs}] != world CRS [{world_grid.crs}]'
            )

        # verify resolution match and compute overlapping extent
        pixel_size = rasters.check_raster_resolution(
            img.transform, lbl.transform if lbl else None
        )
        inter_bbox = rasters.compute_overlap_bounds(
            img.bounds, lbl.bounds if lbl else None
        )

        # focus on tiles inside the intersected extent
        inside = _crop(world_grid, pixel_size, inter_bbox)

        # get image and optionally label reading windows
        return MappedRasterWindows(
            image=_get_windows(world_grid, img.transform, inside),
            label=_get_windows(
                world_grid, lbl.transform if lbl else None, inside
            ),
            grid_id=world_grid.gid,
            tile_shape=world_grid.tile_size,
            tile_overlap=world_grid.tile_overlap,
        )


# ----- private helpers
def _crop(
    world_grid: geo_core.GridLayout,
    pixel_size: tuple[float, float],
    inter_bbox: rasters.BoundingBox,
) -> list[tuple[int, int]]:
    '''Return grid tile indices that intersect the raster extent.'''
    inside: list[tuple[int, int]] = []
    gx, gy = world_grid.origin
    px, py = pixel_size
    min_x = inter_bbox.left - world_grid.tile_size[1] * px
    max_y = inter_bbox.top + world_grid.tile_size[0] * py

    for (x, y) in world_grid.keys():  # keys (x, y) are in pixels
        if (
            gx + x * px < inter_bbox.right and
            gy - y * py > inter_bbox.bottom and
            gx + x * px > min_x and
            gy - y * py < max_y
        ):
            inside.append((x, y))
    return inside


def _get_windows(
    world_grid: geo_core.GridLayout,
    transform: rasters.Affine | None,
    inside_idx: list[tuple[int, int]]
) -> geo_core.RasterWindowDict:
    '''Return window dict for tiles inside the target area.'''
    # get raster reading windows - empty when transform is None
    windows: geo_core.RasterWindowDict = {}
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
