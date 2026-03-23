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
Data preparation pipeline that maps rasters to a grid, builds and
normalizes blocks, splits datasets, and persists a versioned schema.

Public APIs:
    - build_catalogue_test: Run the end-to-end dataset workflow.
'''

# standard imports
import dataclasses
# local imports
import landseg._ingest_dataset.canonical as canonical
import landseg._ingest_dataset.materialized as materialized
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DatasetBuildConfig:
    '''doc'''
    val_test_ratios: tuple[float, float]
    buffer_step: int
    reward_ratios: dict[int, float]     # 0-based
    scoring_alpha: float                # exponent for transforming block counts
    scoring_beta: float                 # reward weight for classes during L1
    max_skew_rate: float
    block_spec: tuple[int, int]         # block_size, block_stride

# -------------------------------Public Function-------------------------------
def build_dataset(
    catalog: canonical.BlocksCatalog,
    config: DatasetBuildConfig,
    output_root: str,
    logger: utils.Logger,
):
    '''doc'''

    logger.log('INFO', 'Test')

    # parse from catalog
    work_catalog = {k: v for k, v in catalog.items() if v['valid_px']}
    catalog_counts = {k: v['class_count'] for k, v in work_catalog.items()}

    # from base grid (no overlap)
    block_size = config.block_spec[0]
    base_catalog = {
        k: v for k, v in work_catalog.items()
        if all(i % block_size == 0 for i in v['loc_col_row']) # both divisible
    }
    base_counts = {k: v['class_count'] for k, v in base_catalog.items()}

    # partition data blocks
    partition_config = materialized.PartitionConfig(
        val_ratio=config.val_test_ratios[0],
        test_ratio=config.val_test_ratios[1],
        buffer_step=config.buffer_step,
        reward_ratios=config.reward_ratios,
        alpha=config.scoring_alpha,
        beta=config.scoring_beta,
        max_skew_rate=config.max_skew_rate
    )
    partitions = materialized.partition_blocks(
        base_counts,
        catalog_counts,
        partition_config,
        logger,
        block_spec=config.block_spec
    )

    # normalize
    train_blocks = [v['file_path'] for k, v in catalog.items() if k in partitions.train]
    all_coords = partitions.train + partitions.val + partitions.test
    all_blocks =  [v['file_path'] for k, v in catalog.items() if k in all_coords]
    materialized.build_normalized_blocks(train_blocks, all_blocks, f'{output_root}/blocks')
