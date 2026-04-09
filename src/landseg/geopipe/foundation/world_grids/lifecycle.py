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
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.world_grids as world_grids
import landseg.utils as utils

# typing aliases
D = list[list[int]]
M = geo_core.GridMeta
CTRL = artifacts.PayloadController[D, M]

# -------------------------------Public Function-------------------------------
def prepare_world_grid(
    logger: utils.Logger,
    grid_fpath: str,
    config: world_grids.GridParameters,
    *,
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

    # payload controller
    ctrl = CTRL(
        grid_fpath,
        schema_id=geo_core.GridLayout.SCHEMA_ID,
        policy=policy
    )
    payload = ctrl.load()
    if payload:
        output_grid = geo_core.GridLayout.from_payload(payload)
        logger.log('INFO', f'World grid loaded successfully; {grid_fpath}')
        return output_grid

    # build if needed
    output_grid = world_grids.build_grid(config)
    payload = output_grid.to_payload()
    ctrl.save(payload)
    logger.log('INFO', f'World grid saved {grid_fpath}')
    return output_grid
