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
import os
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

# -------------------------------Public Function-------------------------------
def partition_blocks(
    foundation_root: str,
    transform_root: str,
    partition_config: PartitionConfig,
    logger: utils.Logger,
) -> None:
    '''doc'''

    # block size (row, col)
    blk_size = (partition_config.block_spec[0], partition_config.block_spec[2])

    # locate catalog JSON files
    dev_catalog = f'{foundation_root}/model_dev/catalog.json'
    test_catalog = f'{foundation_root}/test_holdout/catalog.json'

    # try load test catalog
    ext_test_blks: list[str] = []
    try:
        test = data_partition.parse_catalog(test_catalog, blk_size)
        logger.log('INFO', 'Use external test blocks, no test blocks split')
        ext_test_blks = list(test.valid_file_paths.values())
    except FileNotFoundError:
        logger.log('INFO', 'No external test blocks provided')
        if not partition_config.val_test_ratios[1]:
            logger.log('WARNING', 'No test blocks is to be included')

    # load model dev catalog
    dev = data_partition.parse_catalog(dev_catalog, blk_size)

    # get split blocks
    logger.log('INFO', 'Split data blocks')
    blks_src, start_counts, end_counts = _split(
        dev.base_class_counts,
        dev.valid_class_counts,
        dev.valid_file_paths,
        partition_config,
        ext_test_blks=ext_test_blks
    )
    logger.log('INFO', f'Training blocks: {len(blks_src['train'])} ')
    _log_split_result(start_counts, end_counts, logger) # log more details

    # iterate current traing blocks to get label class counts
    lbl_stats = _count_label(list(blks_src['train'].values()))

    # write JSON artifacts
    utils.write_json(f'{transform_root}/block_source.json', blks_src)
    utils.hash_artifacts(f'{transform_root}/block_source.json')

    utils.write_json(f'{transform_root}/label_stats.json', lbl_stats)
    utils.hash_artifacts(f'{transform_root}/label_stats.json')

def _split(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_class_counts: dict[tuple[int, int], list[int]],
    valid_blocks: dict[tuple[int, int], str],
    config: PartitionConfig,
    *,
    ext_test_blks: list[str] | None
) -> tuple[core.BlockSplitPaths, list[int], list[int]]:
    '''Process wrapper.'''

    # split dataset
    splits = data_partition.stratified_splitter(
        base_class_counts,
        val_ratio=config.val_test_ratios[0],
        test_ratio=(0 if ext_test_blks else config.val_test_ratios[1]),
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
    global_count = numpy.sum(list(base_class_counts.values()), axis=0)
    global_count = [int(x) for x in global_count] # type guard
    blocks_to_score = {
        k: v for k, v in valid_class_counts.items()
        if k in safe_candidates
    }
    ranked_candidates = data_partition.score_blocks(
        global_count,
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

    # block fpaths for each split
    ps: dict[str, list[str]] = {}
    ps['train'] = [valid_blocks[c] for c in splits.train + selected]
    ps['val'] = [valid_blocks[c] for c in splits.val]
    ps['test'] = (ext_test_blks or []) + [valid_blocks[c] for c in splits.test]
    blks_src: core.BlockSplitPaths = {
        'train': _index_fpath(ps['train']),
        'val': _index_fpath(ps['val']),
        'test': _index_fpath(ps['test'])
    }

    # log and return coordinates of each split
    _leak_check(blks_src) # data leakage sanity
    return blks_src, global_count, current_count

def _index_fpath(fpaths: list[str]) -> dict[str, str]:
    '''Index block file paths by their names (no extension).'''

    indexed: dict[str, str] = {}
    for fpath in fpaths:
        filename = os.path.basename(fpath)
        name, _ = os.path.splitext(filename)
        indexed[name] = fpath # name is the same as core.xy_name()
    return indexed

def _leak_check(blks_src: core.BlockSplitPaths) -> None:
    '''Sanity check on data leakage.'''

    leak = set(blks_src['train']) & set(blks_src['val'])
    assert not leak, f'Data leaked between train and val! {leak}'
    leak = set(blks_src['train']) & set(blks_src['test'])
    assert not leak, f'Data leaked between train and test! {leak}'

def _count_label(block_file_list: list[str]) -> dict[str, list[int]]:
    '''Count label classes for each channel.'''

    # iterate current traing blocks to get label class counts
    lbl_stats: dict[str, list[int]] = {}
    for fpath in block_file_list:
        blk_meta = core.DataBlock.load(fpath).meta
        for channel, counts in blk_meta['label_count'].items():
            cls_count = numpy.asarray(counts)
            if channel in lbl_stats:
                lbl_stats[channel] += cls_count
            else:
                lbl_stats[channel] = list(cls_count)
            lbl_stats[channel] = [int(x) for x in lbl_stats[channel]]
    return lbl_stats

def _log_split_result(
    start: list[int],
    current: list[int],
    logger: utils.Logger
) -> None:
    '''Pretty log splitting results.'''

    max_num_str_len = len(f'{max(start + current)}')
    adjust_len = max_num_str_len + max_num_str_len // 3 + 1 # plus commas

    def _per_w_sign(c: float, sign: bool = False) -> str:
        if not sign:
            return f'{abs(c) * 100:.02f}%'
        if c > 0:
            return f'+{abs(c) * 100:.02f}%'
        if c == 0:
            return 'n/a'
        return f'-{abs(c) * 100:.02f}%'

    def _join_per(inputs: list[float], sign: bool = False) -> str:
        return ' '.join(_per_w_sign(c, sign).rjust(adjust_len) for c in inputs)

    def _join_int(inputs: list[int]) -> str:
        return ' '.join(f'{x:,}'.rjust(adjust_len) for x in inputs)

    start_sum = sum(start) or 1.0
    start_per = [float(x / start_sum) for x in start]
    current_sum = sum(current) or 1.0
    current_per = [float(x / current_sum) for x in current]
    count_diff = [x - y for (x, y) in zip(current, start)]
    per_diff = [float(x - y) for (x, y) in zip(current_per, start_per)]

    logger.log('INFO', f'Prev. base label count: {_join_int(start)}')
    logger.log('INFO', f'Curr. base label count: {_join_int(current)}')
    logger.log('INFO', f'            - increase: {_join_int(count_diff)}')
    logger.log('INFO', f'Prev. base label ratio: {_join_per(start_per)}')
    logger.log('INFO', f'Curr. base label ratio: {_join_per(current_per)}')
    logger.log('INFO', f'              - change: {_join_per(per_diff, True)}')
