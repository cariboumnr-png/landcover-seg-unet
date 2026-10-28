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
World grid artifacts lifecycle management.

This module provides functions to prepare, load, and persist world grid
layouts with verification and lifecycle policy handling.

Public APIs:
    - `get_grid_report_fpath`: Return canonical grid report file path.
    - `prepare_world_grid`: Build or load a persisted grid artifact.
'''

# standard imports
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid.builder as builder


# ----- public functions
def prepare_world_grid(
    config: contracts.WorldGridPrepConfig | None = None,
    *,
    load_only: bool = False,
    override_grid_fpath: pathlib.Path | None = None,
) -> tuple[bool, pathlib.Path, geo_core.GridLayout]:
    '''
    Build or load a persisted world grid artifact.

    Args:
        config:
            Configuration object defining grid parameters and paths.
        load_only:
            Whether to strictly load existing grid and fail if missing.
        override_grid_fpath:
            Optional explicit file path to the grid artifact.

    Returns:
        tuple[bool, pathlib.Path, geo_core.GridLayout]:
            Tuple of (is_loaded, grid file path, GridLayout instance).
    '''
    if override_grid_fpath:
        grid_fpath = override_grid_fpath
    else:
        if not config:
            raise ValueError('No config for grid generation is found')
        gid = geo_core.GridLayout.generate_gid(
            config.tile_size, config.tile_stride
        )
        grid_fpath = pathlib.Path(config.output_dpath) / f'{gid}.json'

    policy = (
        artifacts.LifecyclePolicy.LOAD_OR_FAIL
        if load_only
        else artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )
    Ctrl = artifacts.Controller[geo_core.GridPayload]
    ctrl = Ctrl(
        grid_fpath,
        policy,
        schema_id=geo_core.GridLayout.SCHEMA_ID
    )
    payload = ctrl.fetch()

    # load or build
    if payload:
        _grid = geo_core.GridLayout.from_payload(payload)
        is_loaded = True
    else:
        if not config:
            raise ValueError('No config for grid generation is found')
        _grid = builder.build_grid(config)
        payload = _grid.to_payload()
        ctrl.persist(payload)
        is_loaded = False

    return is_loaded, grid_fpath, _grid
