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
Tools for preparing and building world grid layouts.

This module provides functions to construct a `GridLayout` either by
deriving geometry from a reference raster or through manual extent
parameters.

Public APIs:
    - `GridParameters`: Protocol defining grid generation configuration.
    - `build_grid`: Construct a GridLayout from config or raster reference.
'''

# standard imports
from __future__ import annotations
import os
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils


# ----- public types
class GridParameters(typing.Protocol):
    '''Container for grid generation configuration.'''
    @property
    def tile_size(self) -> tuple[int, int]: ...
    @property
    def tile_stride(self) -> tuple[int, int]: ...
    @property
    def ref_fpath(self) -> str | None: ...
    @property
    def crs_string(self) -> str | None: ...
    @property
    def origin(self) -> tuple[float, float] | None: ...
    @property
    def pixel_size(self) -> tuple[float, float] | None: ...
    @property
    def extent_in_crs_units(self) -> tuple[float, float] | None: ...


# ----- public functions
def build_grid(
    mode: typing.Literal['ref', 'manual'] | str,
    config: GridParameters,
) -> geo_core.GridLayout:
    '''
    Build a world grid layout from reference raster or manual parameters.

    Args:
        mode:
            Grid derivation mode ('ref' for reference raster or 'manual'
            for explicit spatial parameters).
        config:
            Configuration object implementing GridParameters protocol.

    Returns:
        geo_core.GridLayout:
            Constructed world grid layout instance.
    '''
    # derive from reference raster
    if mode == 'ref':
        if not (config.ref_fpath and os.path.exists(config.ref_fpath)):
            raise ValueError(f'Invalid reference raster: {config.ref_fpath}')

        with geo_utils.open_rasters(config.ref_fpath) as (src,):
            assert src
            # get transform - pixel size
            transform = src.transform
            px, py = transform.a, abs(transform.e)
            # get bounding box - origin and extent
            l, b, r, t = src.bounds
            # assign to gridspec
            grid_spec = geo_core.GridSpec(
                crs=config.crs_string or str(src.crs),
                origin=(l, t),             # left, top as x, y
                pixel_size=(px, py),       # pixel size in x, y
                tile_size=config.tile_size,
                tile_stride=config.tile_stride,
                grid_extent=(t - b, r - l) # top-bottom as H, right-left as W
            )

    # manually define the extent
    elif mode == 'manual':
        if not config.crs_string:
            raise ValueError('CRS string not provided')
        if not config.origin:
            raise ValueError('Origin not provided')
        if not config.pixel_size:
            raise ValueError('Pixel size not provided')
        if not config.extent_in_crs_units:
            raise ValueError('Extent (in CRS units) not provided')

        grid_spec = geo_core.GridSpec(
            crs=config.crs_string,
            origin=config.origin,
            pixel_size=config.pixel_size,
            tile_size=config.tile_size,
            tile_stride=config.tile_stride,
            grid_extent=config.extent_in_crs_units
        )

    else:
        raise ValueError(f'Invalid extent mode: {mode}')

    output_grid = geo_core.GridLayout(grid_spec)
    return output_grid
