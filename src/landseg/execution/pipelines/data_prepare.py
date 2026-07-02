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
Data preparation (experiment-materialized) pipeline.

Splits raw blocks into train/val(/test), computes train-only band
statistics, normalizes all splits, and emits the final dataset schema.
'''

# standard imports
import datetime
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.transform as transform

def prepare(config: configs.RootConfig):
    '''
    Run the preparation pipeline for an experiment.

    Steps:
    1) Parse current data blocks catalog and schema by config.
    2) Split the blocks into train/val(/test) with configured hydration.
    3) Normalize all blocks using image stats from the train split.
    4) Build schame for downstream consumption.

    Args:
        config: RootConfig with transform settings.
    '''

    # init a TransformLogger
    logger = transform.TransformLogger(
        name='prep',
        log_file=f'{config.execution.exp_root}/prep_report.json',
        enable_file_log=False
    )
    time_stamp = datetime.datetime.now().strftime(c.TF_ISO8601)
    logger.init_summary(f'prepare_{time_stamp}', time_stamp)

    try:
        logger.log_sep()

        # artifact paths
        foundation_paths = artifacts.FoundationPaths(config.foundation.output_dpath)
        transform_paths = artifacts.TransformPaths(config.transform.output_dpath)

        # parse catalog from data foundation
        parsed_catalog = transform.data_blocks_adapter(
            foundation_paths.data_blocks.dev.catalog,
            foundation_paths.data_blocks.dev.schema,
            foundation_paths.data_blocks.test.catalog,
            config=config.transform.catalog
        )

        # datablocks partition
        logger.log('INFO', '[START] Dataset partitioning splits')
        # data transform config aliases
        partition = config.transform.partition
        scoring = config.transform.scoring
        hydration = config.transform.hydration
        # partition config
        partition_config = transform.PartitionParameters(
            val_test_ratios=(partition.val_ratio, partition.test_ratio),
            buffer_step=partition.buffer_step,
            reward_ratios=scoring.reward,
            scoring_alpha=scoring.alpha,
            scoring_beta=scoring.beta,
            max_skew_rate=hydration.max_skew_rate,
            block_spec=config.foundation.grid.tile_specs_tuple
        )
        transform.run_datablocks_partition(
            parsed_catalog,
            transform_paths,
            partition_config,
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger,
        )
        assert logger.summary
        assert logger.summary['data_partition']
        d = logger.summary['data_partition']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Dataset partitioning splits (D_{d:.2f}s)')

        # normalize
        logger.log('INFO', '[START] Block normalization')
        transform.run_normalize_blocks(
            transform_paths,
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger
        )
        assert logger.summary['normalization']
        d = logger.summary['normalization']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Block normalization (D_{d:.2f}s)')

        # build schema
        logger.log('INFO', '[START] Transform schema building')
        transform.build_schema(
            transform_paths,
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger
        )
        assert logger.summary['schema']
        d = logger.summary['schema']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Transform schema building (D_{d:.2f}s)')

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Preparation pipeline failed: {e}', exc_info=True)
        raise e
    finally:
        logger.log_sep()
        logger.close()
