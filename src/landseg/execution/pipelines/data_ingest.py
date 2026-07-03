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

# standard imports
import datetime
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.foundation as foundation

# -------------------------------Public Function-------------------------------
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

    # artifact paths
    paths = artifacts.FoundationPaths(config.foundation.output_dpath)

    # init a FoundationLogger with summary
    logger = foundation.FoundationLogger(
        name='ingest',
        log_file=paths.report,
        enable_file_log=False
    )
    time_stamp = datetime.datetime.now().strftime(c.TF_ISO8601)
    logger.init_summary(f'ingest_{time_stamp}', time_stamp)
    assert logger.summary # typing

    try:
        logger.log_sep()

        # resolve lifecycle policy dynamically
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.execution.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )

        # config aliases
        domain_cfg = config.foundation.domains
        grid_cfg = config.foundation.grid
        datablocks_cfg = config.foundation.datablocks

        # world grid
        logger.log('INFO', '[START] World grid preparation')
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
            policy=policy,
            logger=logger,
        )

        # log to console with duration
        assert logger.summary['world_grid'] # typing: should already populate
        d = logger.summary['world_grid']['duration_sec']
        logger.log('INFO', f'[COMPLETE] World grid preparation (D_{d:.2f}s)')

        # domain maps
        if not domain_cfg.files:
            logger.log('INFO', '[NOTE] No domain knowledge layers provided')
        else:
            logger.log('INFO', '[START] Domain maps preparation')
            gid = world_grid.gid
            domain_config = [
                foundation.DomainBuildingParameters(
                    input_fpath=dm.path,
                    domain_fpath=paths.domains.domain_map_fpath(dm.name),
                    tiles_fpath=paths.domains.mapped_tiles_fpath(dm.name, gid),
                    index_base=dm.index_base,
                    valid_threshold=domain_cfg.valid_threshold,
                    target_variance=domain_cfg.target_variance,
                ) for dm in domain_cfg.files
            ]
            foundation.prepare_domain_maps(
                world_grid,
                domain_config,
                policy=policy,
                logger=logger,
            )

            # log to console with duration
            d = sum(dm['duration_sec'] for dm in logger.summary['domain_maps'])
            logger.log(
                'INFO',
                f'[COMPLETE] Domain maps preparation (D_{d:.2f}s)'
            )

        # build dev data blocks
        logger.log('INFO', '[START] Development data blocks building')
        data_blocks_config = foundation.BlockBuildingParameters(
            stage='dev',
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
            policy=policy,
            logger=logger,
        )

        # log to console with duration
        d = logger.summary['data_blocks']['dev']['duration_sec']
        logger.log(
            'INFO',
            f'[COMPLETE] Development data blocks preparation (D_{d:.2f}s)'
        )

        # build test data blocks - if provided
        if not datablocks_cfg.has_test_data:
            logger.log('INFO', '[NOTE] Test holdout dataset not provided')
        else:
            logger.log('INFO', '[START] Test data blocks building')
            data_blocks_config = foundation.BlockBuildingParameters(
                stage='test',
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
                policy=policy,
                logger=logger,
            )
            assert logger.summary['data_blocks']['test']

            d = logger.summary['data_blocks']['test']['duration_sec']
            logger.log(
                'INFO',
                f'[COMPLETE] Test data blocks preparation (D_{d:.2f}s)'
            )

        # write config JSON sidecar upon successful execution
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    # propagate all exceptions here
    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Ingestion pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close()
