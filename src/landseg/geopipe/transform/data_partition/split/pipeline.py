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
Dataset partitioning pipeline.

Consumes a canonical blocks catalog and produces experiment-specific
dataset splits (train/val/test) based on stratified sampling, spatial
buffering, and class-balance heuristics. Outputs split manifests and
label statistics for downstream normalization and schema generation.
'''

# standard imports
import dataclasses
import os
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.data_partition.split as split

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class PartitionParameters:
    '''Configuration for the dataset partitioning pipeline.'''
    val_test_ratios: tuple[float, float]# val split, test split
    buffer_step: int
    reward_ratios: dict[int, float]     # 0-based
    scoring_alpha: float                # exponent for transforming block counts
    scoring_beta: float                 # reward weight for classes during L1
    max_skew_rate: float
    block_spec: tuple[int, int, int, int]
    # row_size, col_size, row_stride, col_stride

    def __post_init__(self):
        # current we only accept equal row and col sizes and strides
        assert self.block_spec[0] == self.block_spec[1], 'Not a square block'
        assert self.block_spec[2] == self.block_spec[3], 'Not a equal stride'

# -------------------------------Public Function-------------------------------
def create_blocks_partition(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_class_counts: dict[tuple[int, int], list[int]],
    valid_blocks: dict[tuple[int, int], str],
    config: PartitionParameters,
    *,
    ext_test_blks: list[str] | None
) -> tuple[geo_core.BlocksPartition, dict[str, str]]:
    '''Split blocks with spatial safety and class balance.'''

    # split dataset
    splits = split.stratified_splitter(
        base_class_counts,
        val_ratio=config.val_test_ratios[0],
        test_ratio=(0 if ext_test_blks else config.val_test_ratios[1]),
        weight_mode = 'inverse' # default
    )

    # filter candidate blocks for hydration
    safe_candidates = split.filter_safe_tiles(
        list(valid_class_counts.keys()),
        splits.val + splits.test,
        block_size=config.block_spec[0],
        block_stride=config.block_spec[2],
        buffer_steps=config.buffer_step
    )

    # score and rank the safe candidate tiles
    global_count = numpy.sum(list(base_class_counts.values()), axis=0)
    global_count = [int(x) for x in global_count] # type guard
    blocks_to_score = {
        k: v for k, v in valid_class_counts.items()
        if k in safe_candidates
    }
    ranked_candidates = split.score_blocks(
        global_count,
        blocks_to_score,
        reward=tuple(config.reward_ratios.keys()),
        alpha=config.scoring_alpha,
        beta=config.scoring_beta
    )

    # hydrate using the safe candidates
    selected, current_count = split.hydrate_train_split(
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
    blks_src: geo_core.BlocksPartition = {
        'train': _index_fpath(ps['train']),
        'val': _index_fpath(ps['val']),
        'test': _index_fpath(ps['test'])
    }

    # data leakage sanity
    _leak_check(blks_src)
    # return splits and a formatted summary
    summary = {
        '----------Blocks Count': '',
        '       Training blocks': f'{len(blks_src['train'])}',
        '     Validation blocks': f'{len(blks_src['val'])}',
        '           Test blocks': f'{len(blks_src['test'])}',
    }
    summary.update(_summary_split_result(global_count, current_count))
    return blks_src, summary

def _index_fpath(fpaths: list[str]) -> dict[str, str]:
    '''Index block file paths by block name without file extension.'''

    indexed: dict[str, str] = {}
    for fpath in fpaths:
        filename = os.path.basename(fpath)
        name, _ = os.path.splitext(filename)
        indexed[name] = fpath # name is the same as core.xy_name()
    return indexed

def _leak_check(blks_src: geo_core.BlocksPartition) -> None:
    '''Check if any blockshared across train, val, or test splits.'''

    leak = set(blks_src['train']) & set(blks_src['val'])
    assert not leak, f'Data leaked between train and val! {leak}'
    leak = set(blks_src['train']) & set(blks_src['test'])
    assert not leak, f'Data leaked between train and test! {leak}'

def _summary_split_result(
    start: list[int],
    current: list[int],
) -> dict[str, str]:
    '''Pretty-log class count&ratio changes before/after splitting.'''

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

    return {
        '-----Hydration Results': '',
        'Prev. base label count': f'{_join_int(start)}',
        'Curr. base label count': f'{_join_int(current)}',
        '            - increase': f'{_join_int(count_diff)}',
        'Prev. base label ratio': f'{_join_per(start_per)}',
        'Curr. base label ratio': f'{_join_per(current_per)}',
        '              - change': f'{_join_per(per_diff, True)}',
    }
