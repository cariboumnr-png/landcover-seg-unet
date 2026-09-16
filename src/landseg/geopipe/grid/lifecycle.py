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
    - `prepare_world_grid`: Build or load a persisted world grid artifact.
    - `load_grid_from_config`: Load a world grid artifact from configuration.
    - `load_grid_from_fpath`: Load a world grid layout directly from file.
'''

# standard imports
from __future__ import annotations
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid as grid


# ----- typing aliases
D = list[list[int]]
M = geo_core.GridMeta
CTRL = artifacts.PayloadController[D, M]

POLICY = artifacts.LifecyclePolicy.BUILD_IF_MISSING


# ----- private types
class _WorldGridPrepConfig(typing.Protocol):
    '''Config shape to prepare world grid artifacts.'''
    @property
    def mode(self) -> str: ...
    @property
    def params(self) -> grid.GridParameters: ...
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

    ctrl = CTRL(
        grid_fpath,
        schema_id=geo_core.GridLayout.SCHEMA_ID,
        policy=POLICY
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
        _grid = grid.build_grid(config.mode, config.params)
        payload = _grid.to_payload()
        ctrl.save(payload)
        is_loaded = False

    return is_loaded, grid_fpath, _grid


def load_grid_from_config(
    config: _WorldGridPrepConfig,
) -> tuple[str, geo_core.GridLayout]:
    '''
    Load an existing world grid artifact based on input configuration.

    Args:
        config:
            Configuration object defining grid parameters and paths.

    Returns:
        tuple[str, geo_core.GridLayout]:
            Tuple of (grid file path, GridLayout instance).
    '''
    try:
        _, fp, world_grid = prepare_world_grid(config, load_only=True)
        return fp, world_grid
    except ValueError as e:
        raise e # re-raise


def load_grid_from_fpath(fpath: str) -> geo_core.GridLayout:
    '''
    Load a world grid layout directly from a file path.

    Args:
        fpath:
            File path to the serialized grid JSON artifact.

    Returns:
        geo_core.GridLayout:
            Restored GridLayout instance.
    '''
    try:
        _, _, world_grid = prepare_world_grid(override_grid_fpath=fpath)
        return world_grid
    except ValueError as e:
        raise e # re-raise


# ----- private helpers
def _get_grid_fpath(config: _WorldGridPrepConfig) -> str:
    '''Return canonical file path of a world grid artifact.'''
    p = config.params
    gid = geo_core.GridLayout.generate_gid(p.tile_size, p.tile_stride)
    return os.path.join(config.output_dpath, f'{gid}.json')
