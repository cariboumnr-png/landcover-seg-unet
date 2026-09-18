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

# pylint: disable=missing-function-docstring

'''
World grid artifacts lifecycle management.

This module provides functions to prepare, load, and persist world grid
layouts with verification and lifecycle policy handling.

Public APIs:
    - `get_grid_report_fpath`: Return canonical grid report file path.
    - `prepare_world_grid`: Build or load a persisted grid artifact.
'''

# standard imports
from __future__ import annotations
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid.builder as builder


# ----- typing aliases
PayloadCtrl = artifacts.PayloadController[list[list[int]], geo_core.GridMeta]


# ----- private types
class _WorldGridPrepConfig(typing.Protocol):
    '''Config shape to prepare world grid artifacts.'''
    @property
    def mode(self) -> str: ...
    @property
    def params(self) -> builder.GridParameters: ...
    @property
    def output_dpath(self) -> str: ...


# ----- public functions
def prepare_world_grid(
    config: _WorldGridPrepConfig | None = None,
    *,
    load_only: bool = False,
    override_grid_fpath: str | None = None,
) -> tuple[bool, str, geo_core.GridLayout]:
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
        tuple[bool, str, geo_core.GridLayout]:
            Tuple of (is_loaded, grid file path, GridLayout instance).
    '''
    if override_grid_fpath:
        grid_fpath = override_grid_fpath
    else:
        if not config:
            raise ValueError('No config for grid generation is found')
        grid_fpath = _get_grid_fpath(config)

    ctrl = PayloadCtrl(
        grid_fpath,
        schema_id=geo_core.GridLayout.SCHEMA_ID,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )

    payload = ctrl.load()

    # raise if load failed in load_only mode
    if load_only and not payload:
        raise ValueError(f'Loading grid failed: {grid_fpath}')

    # load or build
    if payload:
        _grid = geo_core.GridLayout.from_payload(payload)
        is_loaded = True
    else:
        if not config:
            raise ValueError('No config for grid generation is found')
        _grid = builder.build_grid(config.mode, config.params)
        payload = _grid.to_payload()
        ctrl.save(payload)
        is_loaded = False

    return is_loaded, grid_fpath, _grid


def get_grid_report_fpath(output_dpath: str) -> str:
    '''
    Return canonical file path of the world grid report artifact.

    Args:
        output_dpath:
            Output directory containing world grid artifacts.

    Returns:
        str:
            Full path to the grid_report.json artifact.
    '''
    return geo_core.get_grid_report_fpath(output_dpath)


# ----- private helpers
def _get_grid_fpath(config: _WorldGridPrepConfig) -> str:
    '''Return canonical file path of a world grid artifact.'''
    p = config.params
    gid = geo_core.GridLayout.generate_gid(p.tile_size, p.tile_stride)
    return os.path.join(config.output_dpath, f'{gid}.json')
