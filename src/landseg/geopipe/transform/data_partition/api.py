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
import landseg.geopipe.core as core
import landseg.geopipe.transform.data_partition as data_partition
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

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _SplitResults:
    '''Container for data blocks split results.'''
    train_coords: list[tuple[int, int]]
    val_coords: list[tuple[int, int]]
    test_coords: list[tuple[int, int]]
    original_counts: list[int]
    current_counts: list[int]

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
    ext_test_blks: list[str] = []
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

    # get split blocks
    r = _split(
        dev.base_class_counts,
        dev.valid_class_counts,
        partition_config,
        logger,
        external_test=bool(ext_test_blks)
    )

    # file paths for each split from coords
    blks_src: core.BlockSplitPaths = {
        'train': [dev.valid_file_paths[c] for c in r.train_coords],
        'val': [dev.valid_file_paths[c] for c in r.val_coords],
        'test': ext_test_blks + [dev.valid_file_paths[c] for c in r.test_coords]
    }

    # data leakage sanity
    leak = set(blks_src['train']) & set(blks_src['val'])
    assert not leak, f'Data leaked between train and val! {leak}'
    leak = set(blks_src['train']) & set(blks_src['test'])
    assert not leak, f'Data leaked between train and test! {leak}'

    # label stats dict
    ori = r.original_counts
    cur = r.current_counts
    lbl_stats: core.LabelStats = {
        'original_counts': [int(c) for c in ori],
        'original_proportions': [float(c / sum(ori)) for c in ori],
        'current_counts': cur,
        'current_proportions': [float(c / sum(cur)) for c in cur]
    }

    # write JSON artifacts
    utils.write_json(f'{artifacts_root}/transform/block_source.json', blks_src)
    utils.hash_artifacts(f'{artifacts_root}/transform/block_source.json')

    utils.write_json(f'{artifacts_root}/transform/label_stats.json', lbl_stats)
    utils.hash_artifacts(f'{artifacts_root}/transform/label_stats.json')

def _split(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_class_counts: dict[tuple[int, int], list[int]],
    config: PartitionConfig,
    logger: utils.Logger,
    *,
    external_test: bool
) -> _SplitResults:
    '''Process wrapper.'''

    # split dataset
    splits = data_partition.stratified_splitter(
        base_class_counts,
        val_ratio=config.val_test_ratios[0],
        test_ratio=(0 if external_test else config.val_test_ratios[1]),
        weight_mode = 'inverse' # default
    )

    # filter candidate blocks for hydration
    safe_candidates = data_partition.filter_safe_tiles(
        list(valid_class_counts.keys()),
        splits.val + splits.test,
        block_size=config.block_spec[0],
        block_stride=config.block_spec[1],
        buffer_steps=config.buffer_step
    )

    # score and rank the safe candidate tiles
    base_counts = numpy.array(list(base_class_counts.values()))
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
    selected, current_count = data_partition.hydrate_train_split(
        splits.train_class_count,
        ranked_candidates,
        target_ratios=config.reward_ratios,
        max_skew_rate=config.max_skew_rate
    )

    # log - TBD
    logger.log('INFO', 'Training blocks hydration complete')
    logger.log('INFO', f'Added {len(selected)} additional blocks')

    # return coordinates of each split
    return _SplitResults(
        train_coords=splits.train + selected,
        val_coords=splits.val,
        test_coords=splits.test,
        original_counts=global_class_count,
        current_counts=current_count
    )
