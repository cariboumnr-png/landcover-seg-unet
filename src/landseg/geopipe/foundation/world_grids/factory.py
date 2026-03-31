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
import typing
# third-party import
import rasterio
# local imports
import landseg.geopipe.core as core

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class GridParameters:
    '''Container for grid generation configuration.'''
    mode: typing.Literal['ref', 'aoi', 'tiles']
    crs: str
    ref_fpath: str
    origin: tuple[float, float]
    pixel_size: tuple[float, float]
    grid_extent: tuple[float, float] | None
    grid_shape: tuple[int, int] | None
    tile_specs: tuple[int, int, int, int]

# -------------------------------Public Function-------------------------------
def build_grid(config: GridParameters) -> core.GridLayout:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.

    The grid extent may be derived from a reference raster, an explicit
    AOI, or a tile-based definition.
    '''

    # get gridspec from extent config
    grid_spec = _get_grid_spec(config)

    # build - save - return
    _mode = 'bbox' if config.mode in ['ref', 'aoi'] else 'tiles'
    output_grid = core.GridLayout(_mode, grid_spec)
    return output_grid

# ------------------------------private  function------------------------------
def _get_grid_spec(config: GridParameters) -> core.GridSpec:
    '''Parse grid extent and returns a partially filled `GridSpec`.'''

    # static tile size and overlap
    tile_size = (config.tile_specs[0], config.tile_specs[1])
    tile_overlap = (config.tile_specs[2], config.tile_specs[3])

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
            return core.GridSpec(
                crs=config.crs,
                origin=(l, t),              # left, ,top as x, y
                pixel_size=(px, py),        # pixel size in x, y
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                grid_extent=(t - b, r - l)  # top-bottom as H, right-left as W
            )

    # from aoi (manual)
    elif config.mode  == 'aoi':
        # retrieve inputs and validate
        assert config.grid_extent
        assert all(isinstance(x, float) for x in config.grid_extent)
        return core.GridSpec(
            crs=config.crs,
            origin=config.origin,
            pixel_size=config.pixel_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            grid_extent=config.grid_extent
        )

    # from tiles count (manual)
    elif config.mode  == 'tiles':
        assert config.grid_shape
        assert all(isinstance(x, int) for x in config.grid_shape)
        return core.GridSpec(
            crs=config.crs,
            origin=config.origin,
            pixel_size=config.pixel_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            grid_shape=config.grid_shape
        )
    #
    raise ValueError(f'Invalid extent mode: {config.mode}')
