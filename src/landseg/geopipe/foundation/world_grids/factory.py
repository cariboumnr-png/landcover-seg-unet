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
'''

# standard imports
import dataclasses
import functools
import typing
# third-party import
import rasterio
# local imports
import landseg.geopipe.core as core
import landseg.geopipe.foundation.world_grids as world_grids
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class GridExtentConfig:
    '''Container for grid generation configuration.'''
    mode: typing.Literal['ref', 'aoi', 'tiles']
    crs: str
    ref_fpath: str
    origin: tuple[float, float]
    pixel_size: tuple[float, float]
    grid_extent: tuple[float, float] | None
    grid_shape: tuple[int, int] | None

@dataclasses.dataclass
class GridGenerationConfig:
    '''Container for grid generation configuration.'''
    output_dir: str
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]

# -------------------------------Public Function-------------------------------
def build_world_grid(
    extent_config: GridExtentConfig,
    grid_config: GridGenerationConfig,
    logger: utils.Logger
) -> core.GridLayout:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.

    The grid extent may be derived from a reference raster, an explicit
    AOI, or a tile-based definition.
    '''

    # get a child logger
    logger = logger.get_child('wgrid')

    # get grid id and root dir
    trow, tcol = grid_config.tile_size
    orow, ocol = grid_config.tile_overlap
    gid = f'grid_row_{trow}_{orow}_col_{tcol}_{ocol}'
    outdir = grid_config.output_dir

    # if grid already exist, load and return
    try:
        logger.log('INFO', f'Try to load grid {gid}')
        output_grid = world_grids.load_grid(gid, outdir)
        logger.log('INFO', 'World grid successfully loaded')
        return output_grid
    # otherwise create grid accordingly
    except FileNotFoundError:
        logger.log('INFO', f'World grid {gid} not found, creating from config')

        # get gridspec from extent config
        spec_from_extent = _get_ext(extent_config)
        # finish gridspec from grid profile
        spec = spec_from_extent(
            tile_size=grid_config.tile_size,
            tile_overlap=grid_config.tile_overlap
        )

        # get layout generation mode from extent mode
        # - maybe should separate these
        _mode  = extent_config.mode
        mode = 'bbox' if _mode in ['ref', 'aoi'] else 'tiles'

        # build - save - return
        output_grid = core.GridLayout(mode, spec)
        world_grids.save_grid(output_grid, outdir)
        logger.log('INFO', f'World grid {gid} created and saved at {outdir}')
        return output_grid

# ------------------------------private  function------------------------------
def _get_ext(config: GridExtentConfig,) -> functools.partial[core.GridSpec]:
    '''Parse grid extent and returns a partially filled `GridSpec`.'''

    # from reference raster (auto)
    if config.mode == 'ref':
        # open reference raster
        with rasterio.open(config.ref_fpath) as src:
            # get transform - pixel size
            transform = src.transform
            px, py = transform.a, abs(transform.e)
            # get bounding box - origin and extent
            l, b, r, t = src.bounds
            # assign to gridspec
            return functools.partial(
                core.GridSpec,
                crs=config.crs,
                pixel_size=(px, py),        # pixel size in x, y
                origin=(l, t),              # left, ,top as x, y
                grid_extent=(t - b, r - l)  # top-bottom as H, right-left as W
            )

    # from aoi (manual)
    elif config.mode  == 'aoi':
        # retrieve inputs and validate
        origin = config.origin
        pixel_size = config.pixel_size
        grid_extent = config.grid_extent
        assert grid_extent
        return functools.partial(
            core.GridSpec,
            crs=config.crs,
            pixel_size=(pixel_size[0], pixel_size[1]),
            origin=(origin[0], origin[1]),
            grid_extent=(grid_extent[0], grid_extent[1])
        )

    # from tiles count (manual)
    elif config.mode  == 'tiles':
        origin = config.origin
        pixel_size = config.pixel_size
        grid_shape = config.grid_shape
        assert grid_shape
        return functools.partial(
            core.GridSpec,
            crs=config.crs,
            pixel_size=(pixel_size[0], pixel_size[1]),
            origin=(origin[0], origin[1]),
            grid_shape=(grid_shape[0], grid_shape[1])
        )
    #
    raise ValueError(f'Invalid extent mode: {config.mode}')
