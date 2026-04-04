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

# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.foundation.world_grids as world_grids
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def prepare_world_grid(
    config: world_grids.GridParameters,
    logger: utils.Logger,
    *,
    artifacts_dir: str,
    policy: artifacts.LifecyclePolicy,
) -> geo_core.GridLayout:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.
    '''

    # get a child logger
    logger = logger.get_child('wgrid')

    # get grid id from config
    srow, scol, orow, ocol = config.tile_specs
    gid = f'grid_row_{srow}_{orow}_col_{scol}_{ocol}'

    # load
    output_grid: geo_core.GridLayout | None
    load_status, m, output_grid = world_grids.load_grid(gid, artifacts_dir)
    if load_status: # non-zero status indicates false artifact -> rebuild
        output_grid = None
        logger.log('INFO', f'World grid {gid} loading error: {m}')
    else:
        logger.log('INFO', f'World grid {gid} successfully loaded')

    # policy: build if missing
    if policy is artifacts.LifecyclePolicy.BUILD_IF_MISSING:
        if output_grid:
            return output_grid
    # policy: force rebuild
    if policy is artifacts.LifecyclePolicy.REBUILD:
        pass
    # unsupported policy
    else:
        m = f'Currently unsupported policy: {policy}'
        logger.log('ERROR', m)
        raise NotImplementedError(m)

    output_grid = world_grids.build_grid(config)
    world_grids.save_grid(output_grid, artifacts_dir)
    logger.log('INFO', f'World grid {gid} saved to {artifacts_dir}')
    return output_grid
