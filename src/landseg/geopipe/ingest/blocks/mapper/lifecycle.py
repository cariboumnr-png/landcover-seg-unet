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
Raster-to-grid mapping utilities with lifecycle persistence.

This module provides logic for mapping geospatial raster datasets onto
a predefined grid layout and persisting the resulting window mappings
as artifacts to avoid redundant recomputations.

Public APIs:
    - map_rasters_to_grid: Maps rasters onto grid with caching.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.blocks.mapper.mapper as mapper


# ----- typing aliases
MappingCtrl = artifacts.Controller[dict]


# ----- public functions
def map_rasters_to_grid(
    world_grid: geo_core.GridLayout,
    image_path: str,
    label_path: str | None,
    mapped_windows_path: str,
    *,
    policy: artifacts.LifecyclePolicy,
) -> mapper.MappedRasterWindows:
    '''
    Map raster images and label raster onto a predefined grid layout.

    Ensures that raster tiling over a given world grid is computed once
    and persisted for reuse. If a cached mapping artifact exists at the
    provided path, it is loaded. Otherwise, raster-to-grid alignment is
    computed and serialized via the configured lifecycle policy.

    Args:
        world_grid:
            Target grid layout describing spatial tiling structure.
        image_path:
            File path to source image raster.
        label_path:
            Optional file path to label raster aligned with image.
        mapped_windows_path:
            Path to cached serialized mapping artifact.
        policy:
            Lifecycle policy controlling cache persistence behavior.

    Returns:
        mapper.MappedRasterWindows:
            Container describing mapped grid cells and raster windows.
    '''
    # artifacts controller
    ctrl = MappingCtrl(mapped_windows_path, policy)

    # mapped windows fpath
    payload = ctrl.fetch()
    # build if needed
    if not payload:
        mapped_windows = mapper.map_rasters(world_grid, image_path, label_path)
        payload = {
            'grid_id': mapped_windows.grid_id,
            'tile_shape': list(mapped_windows.tile_shape),
            'image': _canonicalize(mapped_windows.image),
            'label': _canonicalize(mapped_windows.label)
        }
        ctrl.persist(payload)

    # build from payload and return
    mapped_windows = mapper.MappedRasterWindows(
        grid_id=payload['grid_id'],
        tile_shape=tuple(payload['tile_shape']),
        image=_parse(payload['image']),
        label=_parse(payload['label'])
    )
    return mapped_windows


# ----- private helpers
def _canonicalize(mapped_windows: geo_core.RasterWindowDict) -> list[list[int]]:
    '''Create a canonical serialization for mapped windows.'''
    canon: list[list[int]] = []
    for k, w in sorted(mapped_windows.items()):
        canon.append(
            [k[0], k[1], w.col_off, w.row_off, w.width, w.height]
        )
    return canon


def _parse(payload: list[list[int]]) -> geo_core.RasterWindowDict:
    '''Parse window dictionary from serialized payload.'''
    parsed: geo_core.RasterWindowDict = {}
    for c in payload:
        x, y, col_off, row_off, w, h = c
        window = geo_core.RasterWindow(col_off, row_off, w, h) # type: ignore
        parsed[(x, y)] = window
    return parsed
