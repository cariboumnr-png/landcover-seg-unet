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

'''World grid artifacts lifecycle management.'''

# standard imports
import time
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common as common
import landseg.geopipe.foundation.world_grids as world_grids

# typing aliases
D = list[list[int]]
M = geo_core.GridMeta
CTRL = artifacts.PayloadController[D, M]

# -------------------------------Public Function-------------------------------
def prepare_world_grid(
    grid_fpath: str,
    config: world_grids.GridParameters,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: common.FoundationLogger,
) -> geo_core.GridLayout:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.
    '''

    start_time = time.perf_counter()
    # payload controller
    ctrl = CTRL(
        grid_fpath,
        schema_id=geo_core.GridLayout.SCHEMA_ID,
        policy=policy
    )
    payload = ctrl.load()

    loaded_from_disk = False
    # load if present
    if payload:
        _grid = geo_core.GridLayout.from_payload(payload)
        loaded_from_disk = True
    else:
        # build if absent
        _grid = world_grids.build_grid(config)
        payload = _grid.to_payload()
        ctrl.save(payload)

    duration = time.perf_counter() - start_time

    # update structured log
    report: common.WorldGridReport = {
        'grid_id': _grid.gid,
        'status': 'loaded' if loaded_from_disk else 'created_and_loaded',
        'grid_filepath': grid_fpath,
        'crs': str(_grid.crs),
        'pixel_size': _grid.pixel_size,
        'tile_size': _grid.tile_size,
        'tile_overlap': _grid.tile_overlap,
        'duration_sec': duration,
    }
    logger.set_world_grid_report(report)

    return _grid
