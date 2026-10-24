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
    - `build_grid`: Construct a GridLayout from config or raster reference.
'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.rasters as rasters


# ----- Public dataclasses
@dataclasses.dataclass
class GridConfigs:
    '''Grid building configurations'''
    crs_string: str
    origin: tuple[float, float]
    pixel_size: tuple[float, float]
    extent_in_crs_units: tuple[float, float]


# ----- public functions
def build_grid(
    *,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
    mode: typing.Literal['ref', 'manual'] | str,
    ref_fpath: pathlib.Path | None = None,
    manual_specs: GridConfigs | None = None,
) -> geo_core.GridLayout:
    '''
    Build a world grid layout from reference raster or manual parameters.

    Args:
        mode:
            Grid derivation mode ('ref' for reference raster or 'manual'
            for explicit spatial parameters).
        config:
            Grid building configurations.

    Returns:
        geo_core.GridLayout:
            Constructed world grid layout instance.
    '''
    # derive from reference raster
    if mode == 'ref':
        if not (ref_fpath and ref_fpath.exists()):
            raise ValueError(f'Invalid reference raster: {ref_fpath}')

        raster_info = rasters.get_info(ref_fpath)
        l, b, r, t = raster_info.bounds

        grid_spec = geo_core.GridSpec(
            crs=str(raster_info.crs),
            origin=(l, t),                  # left, top as x, y
            pixel_size=raster_info.res,     # pixel size in x, y
            tile_size=tile_size,
            tile_stride=tile_stride,
            grid_extent=(t - b, r - l)      # top-bottom as H, right-left as W
        )

    # manually define the extent
    elif mode == 'manual':
        if not manual_specs:
            raise ValueError('Manual grid specs not provided')

        grid_spec = geo_core.GridSpec(
            crs=manual_specs.crs_string,
            origin=manual_specs.origin,
            pixel_size=manual_specs.pixel_size,
            tile_size=tile_size,
            tile_stride=tile_stride,
            grid_extent=manual_specs.extent_in_crs_units
        )

    else:
        raise ValueError(f'Invalid extent mode: {mode}')

    output_grid = geo_core.GridLayout(grid_spec)
    return output_grid
