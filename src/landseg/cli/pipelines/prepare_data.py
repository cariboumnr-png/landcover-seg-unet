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
    foundation_root = f'{config.foundation.output_dpath}/data_blocks'
    grid = config.foundation.grid
    # data transform
    transform_root = config.transform.output_dpath
    partition = config.transform.partition
    scoring = config.transform.scoring
    hydration = config.transform.hydration

    # datablocks partition
    cfg = transform.PartitionParameters(
        val_test_ratios=(partition.val_ratio, partition.test_ratio),
        buffer_step=partition.buffer_step,
        reward_ratios=scoring.reward,
        scoring_alpha=scoring.alpha,
        scoring_beta=scoring.beta,
        max_skew_rate=hydration.max_skew_rate,
        block_spec=(
            grid.tile_specs.size_row,
            grid.tile_specs.size_col,
            grid.tile_specs.overlap_row,
            grid.tile_specs.overlap_col
        )
    )
    transform.run_datablocks_partition(foundation_root, transform_root, cfg, logger)

    # normalize
    transform.run_normaliza_blocks(transform_root)

    # build schema
    transform.build_schema(transform_root)
