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
import alias
import grid
import utils

# -------------------------------Public Function-------------------------------
def prep_world_grid(
    extent: alias.ConfigType,
    config: alias.ConfigType,
    logger: utils.Logger
) -> tuple[str, grid.GridLayout]:
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
    # config accessor
    extent_cfg = utils.ConfigAccess(extent)
    grid_cfg = utils.ConfigAccess(config)

    # get grid id and root dir
    gid = grid_cfg.get_option('id')
    outdir = grid_cfg.get_option('output_dirpath')

    # if grid already exist, load and return
    try:
        logger.log('INFO', f'Try to load grid {gid} at {outdir}')
        output_grid = grid.load_grid(gid, outdir)
        logger.log('INFO', 'World grid successfully loaded')
        return gid, output_grid
    # otherwise create grid accordingly
    except FileNotFoundError:
        logger.log('INFO', f'World grid {gid} not found, creating from config')
        # get gridspec from extent config
        spec_from_extent = _get_extent(extent_cfg)
        # finish gridspec from grid profile
        spec = spec_from_extent(
            tile_size=(
                grid_cfg.get_option('tile_size', 'row'),
                grid_cfg.get_option('tile_size', 'col')
            ),
            tile_overlap=(
                grid_cfg.get_option('tile_overlap', 'row'),
                grid_cfg.get_option('tile_overlap', 'col')
            )
        )

        # get layout generation mode from extent mode
        # - maybe should separate these
        _mode  = extent_cfg.get_option('mode')
        mode = 'bbox' if _mode in ['ref', 'aoi'] else 'tiles'

        # build - save - return
        output_grid = grid.GridLayout(mode, spec)
        grid.save_grid(gid, output_grid, outdir)
        logger.log('INFO', f'World grid {gid} created and saved at {outdir}')
        return gid, output_grid

# ------------------------------private  function------------------------------
def _get_extent(cfg: utils.ConfigAccess) -> functools.partial[grid.GridSpec]:
    '''Parse grid extent and returns a partially filled `Gridspec`.'''

    # from reference raster (auto)
    if cfg.get_option('mode') == 'ref':
        ref_raster = os.path.join(
            f'{cfg.get_option('inputs', 'dirpath')}',
            f'{cfg.get_option('inputs', 'filename')}'
        )
        if not os.path.exists(ref_raster):
            raise ValueError('Reference raster not provided')
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
                crs=cfg.get_option('crs'),
                pixel_size=(px, py),        # pixel size in x, y
                origin=(l, t),              # left, ,top as x, y
                grid_extent=(t - b, r - l)  # top-bottom as H, right-left as W
            )

    # from aoi (manual)
    elif cfg.get_option('mode') == 'aoi':
        # retrieve inputs and validate
        origin = cfg.get_option('inputs', 'origin')
        pixel_size = cfg.get_option('inputs', 'pixel_size')
        grid_extent = cfg.get_option('inputs', 'grid_extent')
        return functools.partial(
            grid.GridSpec,
            crs=cfg.get_option('crs'),
            pixel_size=(pixel_size[0], pixel_size[1]),
            origin=(origin[0], origin[1]),
            grid_extent=(grid_extent[0], grid_extent[1])
        )

    # from tiles count (manual)
    elif cfg.get_option('mode') == 'tiles':
        origin = cfg.get_option('inputs', 'origin')
        pixel_size = cfg.get_option('inputs', 'pixel_size')
        grid_shape = cfg.get_option('inputs', 'grid_shape')
        return functools.partial(
            grid.GridSpec,
            crs=cfg.get_option('crs'),
            pixel_size=(pixel_size[0], pixel_size[1]),
            origin=(origin[0], origin[1]),
            grid_shape=(grid_shape[0], grid_shape[1])
        )
    #
    raise ValueError(f'Invalid extent mode: {cfg.get_option('mode')}')
