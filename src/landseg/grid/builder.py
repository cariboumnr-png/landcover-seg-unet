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
Tools for preparing and loading world grid layouts.

This module provides the public entry point to build or load a persisted
`GridLayout` from configuration. If the requested grid already exists on
disk, it is loaded; otherwise, the grid specification is derived from the
extent configuration and the grid is created, saved, and returned.

Supported extent modes:
- 'ref'   : derive geometry from a reference raster (bounds, pixel size)
- 'aoi'   : derive from explicit origin, pixel size, and grid extent
- 'tiles' : derive from explicit origin, pixel size, and grid shape

The module depends on:
- `utils.ConfigAccess` for safe, typed config access
- `utils.Logger` for structured logging
- `grid.GridSpec` and `grid.GridLayout` for grid objects
- `grid.load_grid` and `grid.save_grid` for persistence
'''

# standard imports
import functools
import os
# third-party import
import rasterio
# local imports
import landseg.configs as configs
import landseg.grid as grid
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def prep_world_grid(
    extent_config: configs.InputExtentCfg,
    grid_config: configs.PrepGridCfg,
    logger: utils.Logger
) -> grid.GridLayout:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.

    The grid extent may be derived from a reference raster, an explicit
    AOI, or a tile-based definition.

    See Hydra config schema for field-level details:
    - `grid_extent` (modes: 'ref' | 'aoi' | 'tiles')
    - `grid_profile` (ID, root, tile size, overlap)
    '''

    # get a child logger
    logger = logger.get_child('wgrid')

    # get grid id and root dir
    gid = grid_config.id
    outdir = grid_config.output_dirpath

    # if grid already exist, load and return
    try:
        logger.log('INFO', f'Try to load grid {gid}')
        output_grid = grid.load_grid(gid, outdir)
        logger.log('INFO', 'World grid successfully loaded')
        return output_grid
    # otherwise create grid accordingly
    except FileNotFoundError:
        logger.log('INFO', f'World grid {gid} not found, creating from config')

        # get gridspec from extent config
        spec_from_extent = _get_ext(extent_config)
        # finish gridspec from grid profile
        spec = spec_from_extent(
            tile_size=(grid_config.tile_size.row, grid_config.tile_size.col),
            tile_overlap=(grid_config.tile_overlap.row, grid_config.tile_overlap.col)
        )

        # get layout generation mode from extent mode
        # - maybe should separate these
        _mode  = extent_config.mode
        mode = 'bbox' if _mode in ['ref', 'aoi'] else 'tiles'

        # build - save - return
        output_grid = grid.GridLayout(mode, spec)
        grid.save_grid(output_grid, outdir)
        logger.log('INFO', f'World grid {gid} created and saved at {outdir}')
        return output_grid

# ------------------------------private  function------------------------------
def _get_ext(cfg: configs.InputExtentCfg) -> functools.partial[grid.GridSpec]:
    '''Parse grid extent and returns a partially filled `GridSpec`.'''

    # from reference raster (auto)
    if cfg.mode == 'ref':
        ref_raster = os.path.join(cfg.inputs.dirpath, cfg.inputs.filename)
        # open reference raster
        with rasterio.open(ref_raster) as src:
            # get transform - pixel size
            transform = src.transform
            px, py = transform.a, abs(transform.e)
            # get bounding box - origin and extent
            l, b, r, t = src.bounds
            # assign to gridspec
            return functools.partial(
                grid.GridSpec,
                crs=cfg.crs,
                pixel_size=(px, py),        # pixel size in x, y
                origin=(l, t),              # left, ,top as x, y
                grid_extent=(t - b, r - l)  # top-bottom as H, right-left as W
            )

    # from aoi (manual)
    elif cfg.mode  == 'aoi':
        # retrieve inputs and validate
        origin = cfg.inputs.origin
        pixel_size = cfg.inputs.pixel_size
        grid_extent = cfg.inputs.grid_extent
        return functools.partial(
            grid.GridSpec,
            crs=cfg.crs,
            pixel_size=(pixel_size[0], pixel_size[1]),
            origin=(origin[0], origin[1]),
            grid_extent=(grid_extent[0], grid_extent[1])
        )

    # from tiles count (manual)
    elif cfg.mode  == 'tiles':
        origin = cfg.inputs.origin
        pixel_size = cfg.inputs.pixel_size
        grid_shape = cfg.inputs.grid_shape
        return functools.partial(
            grid.GridSpec,
            crs=cfg.crs,
            pixel_size=(pixel_size[0], pixel_size[1]),
            origin=(origin[0], origin[1]),
            grid_shape=(grid_shape[0], grid_shape[1])
        )
    #
    raise ValueError(f'Invalid extent mode: {cfg.mode}')
