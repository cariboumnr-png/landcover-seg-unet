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

'''Domains artifacts lifecycle management.'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
import os
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.domain_maps as domain_maps
import landseg.utils as utils

# typing aliases
D = dict[str, geo_core.DomainTile]
M = geo_core.DomainMeta
DomainCtrl = artifacts.PayloadController[D, M]
MappingCtrl = artifacts.Controller[alias.RasterTileDict]

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DomainBuildingParameters:
    '''Container for domain mapping configurations.'''
    input_fpath: str
    domain_fpath: str
    tiles_fpath: str
    index_base: int
    valid_threshold: float
    target_variance: float

# -------------------------------Public Function-------------------------------
def prepare_domain_maps(
    logger: utils.Logger,
    world_grid: geo_core.GridLayout,
    domain_configs: list[DomainBuildingParameters],
    *,
    policy: artifacts.LifecyclePolicy
) -> None:
    '''
    Prepare and persist domain tile maps for categorical raster(s).

    This function is the public entry point for domain preparation. It
    aligns the provided world grid to each configured domain raster,
    builds a `DomainTileMap` if no artifact exists yet, persists it, and
    returns a dictionary keyed by domain name (filename without suffix).

    Args:
        world_grid: A grid.GridLayout instance describing the tiling to
            use. The domain rasters must share its CRS and pixel size;
            pixel origin alignment is handled internally.
        config: Domain configuration dict. Expected keys
            - 'dirpath': directory containing domain rasters.
            - 'files': list of {'name': str, 'index_base': int}.
            - 'valid_threshold': float in [0, 1], min valid-pixel frac.
            - 'target_variance': float in (0, 1], PCA target EVR.
            - 'output_dirpath': directory for persisted artifacts.
        logger:
            Logger used for structured progress messages.

    Returns:
        dict: A mapping from domain base name to the prepared
            `DomainTileMap`.

    Notes:
    - Existing domain artifacts are loaded and returned without rebuild.
    - New artifacts are saved as JSON payload plus JSON metadata with a
    schema id and integrity hash for compatibility checks.
    '''

    # get a child logger
    logger = logger.get_child('dkmap')

    # read provided domain rasters
    for config in domain_configs:

        # copy the world grid instance
        grid = copy.deepcopy(world_grid)
        name, _ = os.path.splitext(os.path.basename(config.input_fpath))

        # check domain artifacts
        ctrl = DomainCtrl(
            config.domain_fpath,
            schema_id=geo_core.DomainTileMap.SCHEMA_ID,
            policy=policy
        )
        payload = ctrl.load()
        if payload:
            logger.log('INFO', f'Domain {name} loaded successfully')
        else:

            # check mapped tiles before building
            mapped = _prep_mapping(grid, config, policy, logger)

            # build domain map
            domain = domain_maps.build_domain(
                grid.gid,
                mapped,
                config.valid_threshold,
                config.target_variance,
                logger
            )
            payload = domain.to_json_payload()
            ctrl.save(payload)
            logger.log('INFO', f'Domain {name} created successfully')

# ------------------------------private  function------------------------------
def _prep_mapping(
    grid: geo_core.GridLayout,
    config: DomainBuildingParameters,
    policy: artifacts.LifecyclePolicy,
    logger: utils.Logger,
) -> alias.RasterTileDict:
    '''doc'''

    # check mapped tiles before building
    ctrl = MappingCtrl(config.tiles_fpath, policy)
    try:
        mapped = ctrl.fetch()
    except artifacts.ArtifactError as exc:
        logger.log('ERROR', f'Error loading {config.tiles_fpath}: {exc}')
        raise artifacts.ArtifactError from exc
    # create a new mapping if not valid
    if not mapped:
        mapped = domain_maps.map_domain_to_grid(
            grid,
            config.input_fpath,
            config.index_base,
            logger
        )
        ctrl.persist(mapped)
        logger.log('INFO', f'Mapped tiles from {grid.gid} created')

    return mapped
