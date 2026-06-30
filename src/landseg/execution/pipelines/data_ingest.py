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
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.foundation as foundation
import landseg.geopipe.foundation.common as foundation_common

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

    # init a FoundationLogger wrapping console output and mapping JSON report to exp_root
    report_fpath = f'{config.execution.exp_root}/ingest_report.json'
    logger = foundation_common.FoundationLogger(
        name='ingest',
        log_file=report_fpath,
        enable_file_log=False
    )
    run_id = f"ingest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.datetime.now().isoformat()
    logger.init_summary(run_id=run_id, timestamp=timestamp)

    try:
        logger.log_sep()

        # config aliases
        domain_cfg = config.foundation.domains
        grid_cfg = config.foundation.grid
        datablocks_cfg = config.foundation.datablocks

        # artifact paths
        paths = artifacts.FoundationPaths(config.foundation.output_dpath)

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
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger,
        )
        
        # Get grid duration from summary report
        grid_dur = 0.0
        if logger.summary and logger.summary.get('world_grid'):
            grid_dur = logger.summary['world_grid']['duration_sec']
        logger.log('INFO', f'[COMPLETE] World grid preparation (Duration: {grid_dur:.2f}s)')

    # domain maps
        if domain_cfg.files:
            logger.log('INFO', '[START] Domain maps preparation')
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
            # Sum up durations of domain maps from report summary
            total_dur = 0.0
            if logger.summary and logger.summary.get('domain_maps'):
                total_dur = sum(d['duration_sec'] for d in logger.summary['domain_maps'])
            logger.log('INFO', f'[COMPLETE] Domain maps preparation (Duration: {total_dur:.2f}s)')
        else:
            logger.log('INFO', '[NOTE] No domain knowledge layers provided')

        # build dev data blocks
        logger.log('INFO', '[START] Development data blocks building')
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
            stage_key='dev',
        )
        dev_dur = 0.0
        if logger.summary and logger.summary.get('data_blocks') and logger.summary['data_blocks'].get('dev'):
            dev_dur = logger.summary['data_blocks']['dev']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Development data blocks building (Duration: {dev_dur:.2f}s)')

        # build test data blocks - if provided
        if datablocks_cfg.has_test_data:
            logger.log('INFO', '[START] Test data blocks building')
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
                stage_key='test',
            )
            test_dur = 0.0
            if logger.summary and logger.summary.get('data_blocks') and logger.summary['data_blocks'].get('test'):
                test_dur = logger.summary['data_blocks']['test']['duration_sec']
            logger.log('INFO', f'[COMPLETE] Test data blocks building (Duration: {test_dur:.2f}s)')
        else:
            logger.log('INFO', '[NOTE] Test holdout dataset not provided')

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Ingestion pipeline failed: {e}', exc_info=True)
        raise e
    
    finally:
        # close logger
        logger.log_sep()
        logger.close()
