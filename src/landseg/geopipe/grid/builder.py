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
import re
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.rasters as rasters
import landseg.utils.validator as val


# ----- public functions
def build_grid(config: contracts.WorldGridPrepConfig) -> geo_core.GridLayout:
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
    grid_specs = compile_inputs(config)
    return geo_core.GridLayout(grid_specs)


def compile_inputs(config: contracts.WorldGridPrepConfig) -> geo_core.GridSpec:
    '''Validate world grid generation configuration.'''
    if config.mode == 'ref':
        ref_fpath = val.require_file(config.ref_fpath)
        raster_info = rasters.get_info(ref_fpath)
        l, b, r, t = raster_info.bounds
        return geo_core.GridSpec(
            crs=str(raster_info.crs),
            origin=(l, t),                  # left, top as x, y
            pixel_size=raster_info.res,
            tile_size=config.tile_size,
            tile_stride=config.tile_stride,
            grid_extent=(t - b, r - l)      # top-bottom as H, right-left as W
        )
    manual_specs = config.manual_specs
    if not bool(re.fullmatch(r'epsg:\d+', manual_specs.crs_string, re.I)):
        raise val.ValidationError(
            f'CRS string must be EPSG:xxxx, got {manual_specs.crs_string}'
        )
    return geo_core.GridSpec(
        crs=manual_specs.crs_string,
        origin=manual_specs.origin,
        pixel_size=manual_specs.pixel_size,
        tile_size=config.tile_size,
        tile_stride=config.tile_stride,
        grid_extent=manual_specs.extent_in_crs_units
    )
