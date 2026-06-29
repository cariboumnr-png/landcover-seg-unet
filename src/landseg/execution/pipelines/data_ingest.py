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
Data ingestion pipeline.

Prepares the world grid, materializes domain knowledge, and builds
the immutable raw block catalogue for later experiments.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.foundation as foundation
import landseg.utils as utils

def ingest(config: configs.RootConfig):
    '''
    Run the ingestion pipeline.

    Steps:
    1) Build or load the world grid.
    2) Prepare domain knowledge aligned to the grid.
    3) Build raw `.npz` data blocks, and update `catalog.json` and
    `schema.json`.

    Args:
        config: RootConfig with foundation settings.
    '''

    # init a logger
    logger = utils.Logger('ingest', f'{config.execution.exp_root}/ingest.log')
    logger.log_sep()

    # config aliases
    domain_cfg = config.foundation.domains
    grid_cfg = config.foundation.grid
    datablocks_cfg = config.foundation.datablocks

    # artifact paths
    paths = artifacts.FoundationPaths(config.foundation.output_dpath)

    # world grid
    logger.log('INFO', 'START/ Create or load world grid')
    grid_config = foundation.GridParameters(
        mode=grid_cfg.mode, # type: ignore
        crs=grid_cfg.crs,
        ref_fpath=grid_cfg.extent.filepath,
        origin=grid_cfg.extent.origin,
        pixel_size=grid_cfg.extent.pixel_size,
        grid_extent=grid_cfg.extent.grid_extent,
        grid_shape=grid_cfg.extent.grid_shape,
        tile_specs=grid_cfg.tile_specs_tuple,
    )
    world_grid = foundation.prepare_world_grid(
        paths.grids.fpath(grid_cfg.tile_specs_tuple),
        grid_config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger,
    )
    logger.log('INFO', f'COMPLETE/ World grid {world_grid.gid} created/loaded')

    # domain maps
    if domain_cfg.files:
        logger.log('INFO', 'START/ Create or load domain knowledge layers')
        domain_config = [
            foundation.DomainBuildingParameters(
            input_fpath=d.path,
            domain_fpath=paths.domains.domain_map_fpath(d.name),
            tiles_fpath=paths.domains.mapped_tiles_fpath(d.name, world_grid.gid),
            index_base=d.index_base,
            valid_threshold=domain_cfg.valid_threshold,
            target_variance=domain_cfg.target_variance,
            ) for d in domain_cfg.files
        ]
        foundation.prepare_domain_maps(
            world_grid,
            domain_config,
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger,
        )
        logger.log('INFO', 'COMPLETE/ Domain knowledge layers created/loaded')
    else:
        logger.log('INFO', 'NOTE/ No domain knowledge layers provided')

    # build dev data blocks
    logger.log('INFO', 'START/ Build or update development data blocks')
    data_blocks_config = foundation.BlockBuildingParameters(
        image_fpath=datablocks_cfg.filepaths.dev_image,
        label_fpath=datablocks_cfg.filepaths.dev_label,
        data_config_fpath=datablocks_cfg.filepaths.config,
        dem_pad=datablocks_cfg.general.image_dem_pad,
        ignore_index=datablocks_cfg.general.ignore_index,
    )
    foundation.run_blocks_building(
        world_grid,
        paths.data_blocks.dev,
        data_blocks_config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger,
    )
    logger.log('INFO', 'COMPLETE/ Development data blocks built/updated')

    # build test data blocks - if provided
    if datablocks_cfg.has_test_data:
        logger.log('INFO', 'START/ Build or update test-holdout data blocks')
        data_blocks_config = foundation.BlockBuildingParameters(
            image_fpath=datablocks_cfg.filepaths.test_image,
            label_fpath=datablocks_cfg.filepaths.test_label,
            data_config_fpath=datablocks_cfg.filepaths.config,
            dem_pad=datablocks_cfg.general.image_dem_pad,
            ignore_index=datablocks_cfg.general.ignore_index,
        )
        foundation.run_blocks_building(
            world_grid,
            paths.data_blocks.test,
            data_blocks_config,
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger,
        )
        logger.log('INFO', 'COMPLETE/ Test-holdout data blocks built/updated')
    else:
        logger.log('INFO', 'NOTE/ Test holdout dataset not provided')

    # close logger
    logger.log_sep()
    logger.close()
