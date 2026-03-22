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
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg._ingest_dataset.canonical as canonical
import landseg._ingest_dataset.materialized as materialized
import landseg.utils as utils

def materialize_dataset_test(
    base_grid: core.GridLayoutLike,
    config: configs.RootConfig,
    logger: utils.Logger,
):
    '''doc'''

    logger.log('INFO', 'Test')
    root = './experiment/artifacts/data_cache/branch_test'

    # from catalog.json
    catalog = canonical.BlocksCatalog.from_json(f'{root}/fit/catalog.json')
    catalog = {k: v for k, v in catalog.items() if v['valid_px']}
    catalog_counts = {k: v['class_count'] for k, v in catalog.items()}

    # from base grid (no overlap)
    base_coords = list(base_grid.keys())
    base_catalog = {
        k: v for k, v in catalog.items()
        if tuple(v['loc_col_row']) in base_coords
    }
    base_counts = {k: v['class_count'] for k, v in base_catalog.items()}

    # partition data blocks
    partition_config = materialized.PartitionConfig(
        val_ratio=0.1,
        test_ratio=0.1,
        buffer_step=1,
        reward_ratios={2: 5.0, 4: 5.0},
        alpha=config.prep.data.scoring.alpha,
        beta=config.prep.data.scoring.beta,
        max_skew_rate=5.0
    )
    partitions = materialized.partition_blocks(
        base_counts,
        catalog_counts,
        partition_config,
        logger,
        block_spec=(256, 128)
    )

    # normalize
    train_blocks = [v['file_path'] for k, v in catalog.items() if k in partitions.train]
    all_coords = partitions.train + partitions.val + partitions.test
    all_blocks =  [v['file_path'] for k, v in catalog.items() if k in all_coords]
    materialized.build_normalized_blocks(
        train_blocks,
        all_blocks,
        f'{root}/test_blocks'
    )
