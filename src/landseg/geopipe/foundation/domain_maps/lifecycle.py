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
import landseg.geopipe.core as geo_core
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.domain_maps as domain_maps
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DomainBuildingParameters:
    '''Container for domain mapping configurations.'''
    src_path: str
    index_base: int
    valid_threshold: float
    target_variance: float

# -------------------------------Public Function-------------------------------
def prepare_domain_maps(
    world_grid: geo_core.GridLayout,
    domain_configs: list[DomainBuildingParameters],
    logger: utils.Logger,
    *,
    artifacts_dir: str,
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
        name, _ = os.path.splitext(os.path.basename(config.src_path))

        # check domain artifacts
        domain = _domain_map_status(name, artifacts_dir, logger, policy)
        if not domain:

            # check mapped tiles before building
            tiles_fpath = f'{artifacts_dir}/{name}_tiles_{grid.gid}.npz'
            mapped = _mapping_status(tiles_fpath, logger, policy)
            # create a new mapping if not valid
            if not mapped:
                mapped = domain_maps.map_domain_to_grid(
                    grid,
                    config.src_path,
                    config.index_base,
                    logger
                )
                logger.log('INFO', f'Mapped tiles from {grid.gid} created')
                artifacts.write_dict_npz_hash(tiles_fpath, mapped)

            # build domain map
            domain = domain_maps.build_domain(
                grid.gid,
                mapped,
                config.valid_threshold,
                config.target_variance,
                logger
            )
            payload = domain.to_json_payload()
            domain_map_fpath = f'{artifacts_dir}/{name}.json'
            domain_meta_fpath = f'{artifacts_dir}/{name}_meta.json'
            #
            data = payload.pop('data')
            artifacts.write_json_hash(domain_map_fpath, data)
            artifacts.write_json_hash(domain_meta_fpath, payload)
#
def _mapping_status(
    tiles_fpath: str,
    logger: utils.Logger,
    policy: artifacts.LifecyclePolicy
) -> alias.RasterTileDict | None:
    '''doc'''

    # load tiles artifact
    mapped_tiles: alias.RasterTileDict | None
    load_status, m, mapped_tiles = artifacts.load_dict_npz_hash(tiles_fpath)
    if load_status: # non-zero status indicates false artifact -> rebuild
        mapped_tiles = None
        logger.log('INFO', f'Mapped tiles loading error: {m}')
    else:
        logger.log('INFO', f'Mapped tiles loaded from {tiles_fpath}')

    # policy: build if missing
    if policy is artifacts.LifecyclePolicy.BUILD_IF_MISSING:
        return mapped_tiles
    # policy: force rebuild
    if policy is artifacts.LifecyclePolicy.REBUILD:
        return None
    # unsupported policy
    m = f'Currently unsupported policy: {policy}'
    logger.log('ERROR', m)
    raise NotImplementedError(m)

def _domain_map_status(
    name: str,
    artifacts_dir: str,
    logger: utils.Logger,
    policy: artifacts.LifecyclePolicy
) -> geo_core.DomainTileMap | None:
    '''doc'''

    # load domain
    domain: geo_core.DomainTileMap | None
    status, msg, domain = domain_maps.load_domain(name, artifacts_dir)
    if status:
        domain = None
        logger.log('INFO', f'Errors encountered during loading: {msg}')
    else:
        assert isinstance(domain, geo_core.DomainTileMap) # typing guard
        logger.log('INFO', f'Domain {name} loaded successfully')

    # policy: build if missing
    if policy is artifacts.LifecyclePolicy.BUILD_IF_MISSING:
        return domain
    # policy: force rebuild
    if policy is artifacts.LifecyclePolicy.REBUILD:
        return None
    # unsupported policy
    m = f'Currently unsupported policy: {policy}'
    logger.log('ERROR', m)
    raise NotImplementedError(m)
