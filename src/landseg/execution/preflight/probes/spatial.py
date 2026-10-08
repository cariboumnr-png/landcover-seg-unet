# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Spatial grid and coordinate reference system diagnostic probes.

Inspects spatial projection configurations, block dimension contracts,
and world grid artifacts prior to pipeline execution.

Public APIs:
    - `probe_spatial`: Run spatial coordinate and grid checks.
'''

# standard imports
import os
# local imports
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def spatial_reference(
    config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Reports the world grid spatial definition source.'''
    grid_cfg = config.data.world_grid

    if grid_cfg.mode == 'manual':
        return schema.ProbeResult(
            pid='world_grid_reference',
            category='Spatial',
            status=schema.ProbeStatus.PASS,
            message='Manual spatial definitions below',
        )

    ref_fpath = grid_cfg.params.ref_fpath
    return schema.ProbeResult(
        pid='world_grid_reference',
        category='Spatial',
        status=(
            schema.ProbeStatus.PASS
            if ref_fpath is not None and os.path.isfile(ref_fpath)
            else schema.ProbeStatus.FAIL
        ),
        message=f'Reference raster found at: {ref_fpath}',
        details={'path': ref_fpath},
    )


def crs_info(
    config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Reports the target coordinate reference system.'''
    grid_cfg = config.data.world_grid
    if grid_cfg.mode == 'ref':
        message = 'Target CRS is defined by the reference raster'
    else:
        message = f'Target CRS: {grid_cfg.params.crs_string}'

    return schema.ProbeResult(
        pid='crs',
        category='Spatial',
        status=_ref_mode_status(config),
        message=message,
    )


def pixel_size(
    config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Reports the world grid pixel size.'''
    grid_cfg = config.data.world_grid
    if grid_cfg.mode == 'ref':
        message = 'Pixel size is defined by the reference raster'
    else:
        message = f'Pixel size: {grid_cfg.params.pixel_size}'

    return schema.ProbeResult(
        pid='pixel_size',
        category='Spatial',
        status=_ref_mode_status(config),
        message=message,
    )


def grid_extent(
    config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Reports the world grid extent.'''
    grid_cfg = config.data.world_grid
    if grid_cfg.mode == 'ref':
        message = 'Grid extent is defined by the reference raster'
    else:
        message = f'Grid extent: {grid_cfg.params.extent_in_crs_units}'

    return schema.ProbeResult(
        pid='grid_extent',
        category='Spatial',
        status=_ref_mode_status(config),
        message=message,
    )


def grid_origin(
    config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Reports the world grid origin.'''
    grid_cfg = config.data.world_grid
    if grid_cfg.mode == 'ref':
        message = 'Grid origin is defined by the reference raster'
    else:
        message = f'Grid origin: {grid_cfg.params.origin}'

    return schema.ProbeResult(
        pid='grid_origin',
        category='Spatial',
        status=_ref_mode_status(config),
        message=message,
    )


def grid_specs(
    config: configs.RootConfig,
) -> schema.ProbeResult:
    '''Reports the configured tile dimensions and stride.'''
    bx, by = config.data.world_grid.params.tile_size
    sx, sy = config.data.world_grid.params.tile_stride

    return schema.ProbeResult(
        pid='block_specifications',
        category='Spatial',
        status=schema.ProbeStatus.PASS,
        message=f'{bx}x{by} px tile with {sx}x{sy} px stride',
    )


def _ref_mode_status(config: configs.RootConfig) -> schema.ProbeStatus:
    if config.data.world_grid.mode == 'ref':
        status = schema.ProbeStatus.SKIP
    else:
        status = schema.ProbeStatus.PASS
    return status
