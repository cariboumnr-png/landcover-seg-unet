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
Raster-to-grid mapping utilities.

This module provides logic for mapping geospatial raster datasets onto
a predefined grid layout and persisting the resulting window mappings
for reuse.

It supports:
- Computation of raster window alignment over a spatial grid
- Optional handling of paired image/label rasters
- Serialization of mappings into a canonical cached format
- Loading cached mappings to avoid recomputation

The module is designed to ensure deterministic, reusable raster tiling
across dataset builds and schema generation pipelines.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.mapper as mapper
import landseg.utils as utils

# typing aliases
MappingCtrl = artifacts.Controller[dict]

# -------------------------------Public Function-------------------------------
def map_rasters_to_grid(
    world_grid: geo_core.GridLayout,
    image_path: str,
    label_path: str | None,
    mapped_windows_path: str,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: utils.Logger,
) -> mapper.MappedRasterWindows:
    '''
    Map raster images and label raster onto a predefined grid layout.

    This function is responsible for ensuring that raster tiling over a
    given world grid is computed once and then persisted for reuse. If a
    cached mapping artifact exists at the provided path, it is loaded and
    deserialized. Otherwise, raster-to-grid alignment is computed using
    `mapper.map_rasters`, and the resulting window layout is serialized
    and stored via the configured lifecycle policy.

    The mapping associates each grid cell with a raster window (col/row
    offsets and spatial extent), separately for image and optional label
    rasters.

    Args:
        world_grid:
            Target grid layout describing spatial tiling structure.
        image_path:
            File path to the source image raster.
        label_path:
            Optional file path to a label raster aligned with the image.
            May be None for unlabeled datasets.
        mapped_windows_path:
            Path to the cached serialized mapping artifact.
        policy:
            Lifecycle policy controlling cache persistence behavior.
        logger:
            Logger used for tracing and cache status reporting.

    Returns:
        A `MappedRasterWindows` object describing the mapping between
        grid cells and raster windows for both image and label (if
        present).
    '''

    # artifacts controller
    ctrl = MappingCtrl(mapped_windows_path, policy)

    # mapped windows fpath
    logger.log('INFO', f'Try to load mapped windows from {world_grid.gid}')
    payload = ctrl.fetch()

    # build if needed
    if not payload:
        mapped_windows = mapper.map_rasters(
            world_grid,
            image_path,
            label_path,
            logger=logger
        )
        payload = {
            'grid_id': mapped_windows.grid_id,
            'tile_shape': list(mapped_windows.tile_shape),
            'image': _canonicalize(mapped_windows.image),
            'label': _canonicalize(mapped_windows.label)
        }
        ctrl.persist(payload)
        logger.log('INFO', f'Mapped windows from {world_grid.gid} created')

    else:
        mapped_windows = mapper.MappedRasterWindows(
            grid_id=payload['grid_id'],
            tile_shape=tuple(payload['tile_shape']),
            image=_parse(payload['image']),
            label=_parse(payload['label'])
        )

    # return
    return mapped_windows

def _canonicalize(mapped_windows: alias.RasterWindowDict) -> list[list[int]]:
    '''Create a canonical serialization for mapped windows.'''

    canon: list[list[int]] = []
    for k, w in sorted(mapped_windows.items()):
        canon.append(
            [k[0], k[1], w.col_off, w.row_off, w.width, w.height]
        )
    return canon

def _parse(payload: list[list[int]]) -> alias.RasterWindowDict:
    '''Parse window dict from payload.'''

    parsed: alias.RasterWindowDict = {}
    for c in payload:
        x, y, col_off, row_off, w, h = c
        window = alias.RasterWindow(col_off, row_off, w, h) # type: ignore
        parsed[(x, y)] = window
    return parsed
