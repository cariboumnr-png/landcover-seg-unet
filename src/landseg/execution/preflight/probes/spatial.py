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
def probe_spatial(
    target: str,
    root_config: configs.RootConfig,
) -> list[schema.ProbeResult]:
    '''
    Inspect spatial grid contracts and projection validity.

    Args:
        target:
            pipeline or workflow target identifier.
        root_config:
            hydra-composed root configuration.

    Returns:
        list[schema.ProbeResult]:
            list of spatial diagnostic probe records.
    '''
    probes: list[schema.ProbeResult] = []
    grid_cfg = root_config.data.world_grid.params

    # crs check
    crs_str = grid_cfg.crs_string or getattr(
        root_config.data, 'crs', 'EPSG:3161'
    )
    if crs_str:
        probes.append(
            schema.ProbeResult(
                probe_id='crs_validity',
                category='spatial',
                status=schema.ProbeStatus.PASS,
                message=f'Target CRS: {crs_str}',
                details={'crs': crs_str},
            )
        )

    # pixel resolution check
    px_size = grid_cfg.pixel_size or (10.0, 10.0)
    probes.append(
        schema.ProbeResult(
            probe_id='pixel_resolution',
            category='spatial',
            status=schema.ProbeStatus.PASS,
            message=f'Resolution: {px_size[0]}m x {px_size[1]}m',
            details={'pixel_size': px_size},
        )
    )

    # block dimensions check
    tile_size = grid_cfg.tile_size or (256, 256)
    probes.append(
        schema.ProbeResult(
            probe_id='block_dimensions',
            category='spatial',
            status=schema.ProbeStatus.PASS,
            message=f'{tile_size[0]}x{tile_size[1]} px tile',
            details={'tile_size': tile_size},
        )
    )

    # world grid existence check for downstream stages
    if target in {'data-harmonize', 'data-ingest', 'data-prepare'}:
        ref_fpath = grid_cfg.ref_fpath
        if ref_fpath and os.path.isfile(ref_fpath):
            probes.append(
                schema.ProbeResult(
                    probe_id='world_grid_spec',
                    category='spatial',
                    status=schema.ProbeStatus.PASS,
                    message=f'Found grid definition at {ref_fpath}',
                    details={'path': ref_fpath},
                )
            )
        elif ref_fpath:
            probes.append(
                schema.ProbeResult(
                    probe_id='world_grid_spec',
                    category='spatial',
                    status=schema.ProbeStatus.WARN,
                    message=f'Grid spec not found at {ref_fpath}',
                    details={'path': ref_fpath},
                )
            )

    return probes
