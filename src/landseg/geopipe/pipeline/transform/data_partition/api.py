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
Dataset partioning pipeline.
'''

# standard imports
import dataclasses
# third-party imports
import numpy
# local imports
import landseg.geopipe.pipeline.common.alias as alias
import landseg.geopipe.pipeline.transform.data_partition as data_partition
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class PartitionConfig:
    '''Partition pipeline configuration.'''
    val_test_ratios: tuple[float, float]# val split, test split
    buffer_step: int
    reward_ratios: dict[int, float]     # 0-based
    scoring_alpha: float                # exponent for transforming block counts
    scoring_beta: float                 # reward weight for classes during L1
    max_skew_rate: float
    block_spec: tuple[int, int, int, int]  # row_size, row_stride, col_size, col_stride

    def __post_init__(self):
        # current we only accept equal row and col sizes and strides
        assert self.block_spec[0] == self.block_spec[2], 'Not a square block'
        assert self.block_spec[1] == self.block_spec[3], 'Not a equal stride'

# -------------------------------Public Function-------------------------------
def partition_blocks(
    artifacts_root: str,
    partition_config: PartitionConfig,
    logger: utils.Logger,
) -> None:
    '''doc'''

    # block size (row, col)
    blk_size = (partition_config.block_spec[0], partition_config.block_spec[2])

    # locate catalog JSON files
    dev_catalog = f'{artifacts_root}/foundation/model_dev/catalog.json'
    test_catalog = f'{artifacts_root}/foundation/test_holdout/catalog.json'

    # try load test catalog
    ext_test_blks = []
    try:
        test = data_partition.parse_catalog(test_catalog, blk_size)
        logger.log('INFO', 'Use external test blocks, no test blocks split')
        ext_test_blks = list(test.valid_file_paths.values())
    except FileNotFoundError:
        logger.log('INFO', 'No external test blocks provided')
        if not partition_config.val_test_ratios[1]:
            logger.log('WARNING', 'No test blocks is to splitted')

    # load model dev catalog
    dev = data_partition.parse_catalog(dev_catalog, blk_size)

    # get split as block coordinates
    splits = _process(
        dev.base_class_counts,
        dev.valid_class_counts,
        partition_config,
        logger,
        external_test=bool(ext_test_blks)
    )

    # file paths for each split from coords
    train = [dev.valid_file_paths[c] for c in splits[0]]
    val = [dev.valid_file_paths[c] for c in splits[1]]
    test = ext_test_blks + [dev.valid_file_paths[c] for c in splits[2]]
    # data leakage sanity
    leak = set(train) & set(val)
    assert not leak, f'Data leaked between train and val! {leak}'
    leak = set(train) & set(test)
    assert not leak, f'Data leaked between train and test! {leak}'

    # write the splits and metadata to JSON
    utils.write_json(f'{artifacts_root}/transform/train_blocks.json', train)
    utils.write_json(f'{artifacts_root}/transform/val_blocks.json', val)
    utils.write_json(f'{artifacts_root}/transform/test_blocks.json', test)

    # hash JSON artifacts
    utils.hash_artifacts(f'{artifacts_root}/transform/train_blocks.json')
    utils.hash_artifacts(f'{artifacts_root}/transform/val_blocks.json')
    utils.hash_artifacts(f'{artifacts_root}/transform/test_blocks.json')

def _process(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_class_counts: dict[tuple[int, int], list[int]],
    config: PartitionConfig,
    logger: utils.Logger,
    *,
    external_test: bool
) -> tuple[alias.CoordsList, alias.CoordsList, alias.CoordsList]:
    '''Process wrapper.'''

    # split dataset
    base_counts = numpy.array(list(base_class_counts.values()))
    splits = data_partition.stratified_splitter(
        base_counts,
        val_ratio=config.val_test_ratios[0],
        test_ratio=(0 if external_test else config.val_test_ratios[1]),
        weight_mode = 'inverse' # default
    )

    # filter candidate files (remove overlaps from val+test blocks)
    catalog_coords = list(valid_class_counts.keys())
    base_coords = list(base_class_counts.keys())
    exclude_coords = [base_coords[i] for i in [*splits.val, *splits.test]]
    safe_candidates = data_partition.filter_safe_tiles(
        catalog_coords,
        exclude_coords,
        block_size=config.block_spec[0],
        block_stride=config.block_spec[1],
        buffer_steps=config.buffer_step
    )

    # score and rank the safe candidate tiles
    global_class_count = numpy.sum(base_counts, axis=0)
    blocks_to_score = {
        k: v for k, v in valid_class_counts.items()
        if k in safe_candidates
    }
    ranked_candidates = data_partition.score_blocks(
        global_class_count,
        blocks_to_score,
        reward=tuple(config.reward_ratios.keys()),
        alpha=config.scoring_alpha,
        beta=config.scoring_beta
    )

    # hydrate using the safe candidates
    train_class_count = list(numpy.sum(base_counts[list(splits.train)], axis=0))
    selected, cur_count = data_partition.hydrate_train_split(
        train_class_count,
        ranked_candidates,
        target_ratios=config.reward_ratios,
        max_skew_rate=config.max_skew_rate
    )

    # log - TBD
    logger.log('INFO', 'Training blocks hydration complete')
    logger.log('INFO', f'Added {len(selected)} additional blocks')
    logger.log('INFO', f'Previous class count: {[int(x) for x in train_class_count]}')
    logger.log('INFO', f'Current class count: {[int(x) for x in cur_count]}')
    logger.log('INFO', f'Class count increased: {numpy.array(cur_count) / train_class_count}')

    # return coordinates of each split
    return (
        [base_coords[i] for i in splits.train] + selected,
        [base_coords[i] for i in splits.val],
        [base_coords[i] for i in splits.test],
    )
