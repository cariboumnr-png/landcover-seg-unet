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
import os
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.foundation.domain_maps as domain_maps
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def prepare_domain_maps(
    world_grid: geo_core.GridLayout,
    domain_configs: list[domain_maps.DomainBuildingParameters],
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
        # get filepath and filename without extension
        name, _ = os.path.splitext(os.path.basename(config.src_path))

        # rebuild if missing or outdated
        if policy is artifacts.LifecyclePolicy.REBUILD_IF_STALE:
            msg = '' # default status message
            # try load the domain JSON
            try:
                logger.log('INFO', f'Loading domain {name}')
                dom = domain_maps.load_domain(name, artifacts_dir, logger)
                # assess domain JSON status
                if not grid.tile_overlap in dom.blk_overlaps:
                    msg = 'Found new tile stride from the input grid'
                if dom.blk_size != grid.tile_size:
                    msg = 'Domain was mapped to a grid of different tiles size'
            except FileNotFoundError:
                msg = 'Domain JSON not found'

            if bool(msg):
                logger.log('INFO', msg)
                dom = domain_maps.build_domain(grid, config, logger)
                domain_maps.save_domain(name, dom, artifacts_dir)
                logger.log('INFO', f'Domain {name} created/updated')
            else:
                logger.log('INFO', f'Domain {name} loaded successfully')

        # force rebuild
        elif policy is artifacts.LifecyclePolicy.REBUILD:
            dom = domain_maps.build_domain(grid, config, logger)
            domain_maps.save_domain(name, dom, artifacts_dir)
            logger.log('INFO', f'Domain {name} rebuilt')

        # unsupported policy
        else:
            msg = f'Currently unsupported policy: {policy}'
            logger.log('ERROR', msg)
            raise NotImplementedError(msg)
