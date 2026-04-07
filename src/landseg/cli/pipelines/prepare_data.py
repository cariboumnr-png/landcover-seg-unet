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

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.transform as transform
import landseg.utils as utils

def prepare(config: configs.RootConfig):
    '''
    Run the preparation pipeline for an experiment.

    Outputs:
    - `block_source.json`, `label_stats.json`
    - `image_stats.json`, normalized `.npz` per split, `block_splits.json`
    - `schema.json` referencing only normalized artifacts

    Args:
        config: RootConfig with transform settings.
    '''

    # init a logger
    logger = utils.Logger('prep', f'{config.exp_root}/prep.log')

    # config aliases
    # data foundation
    grid = config.foundation.grid
    # data transform
    partition = config.transform.partition
    scoring = config.transform.scoring
    hydration = config.transform.hydration

    # artifact paths
    foundation_paths = artifacts.FoundationPaths(config.foundation.output_dpath)
    transform_paths = artifacts.TransformPaths(config.transform.output_dpath)

    # datablocks partition
    # parse catalog
    parsed_catalog = transform.parse_catalog(
        foundation_paths.data_blocks.dev.catalog,
        foundation_paths.data_blocks.dev.schema,
        foundation_paths.data_blocks.test.catalog,
        valid_px_threshold=0.8
    )
    # partition config
    partition_config = transform.PartitionParameters(
        val_test_ratios=(partition.val_ratio, partition.test_ratio),
        buffer_step=partition.buffer_step,
        reward_ratios=scoring.reward,
        scoring_alpha=scoring.alpha,
        scoring_beta=scoring.beta,
        max_skew_rate=hydration.max_skew_rate,
        block_spec=grid.tile_specs_tuple
    )
    transform.run_datablocks_partition(
        logger,
        transform_paths,
        parsed_catalog,
        partition_config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
    )

    # normalize
    transform.run_normaliza_blocks(
        transform_paths,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )

    # build schema
    transform.build_schema(
        transform_paths,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )
