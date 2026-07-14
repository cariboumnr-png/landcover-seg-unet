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
        row_size, col_size, row_stride, col_stride = self.block_spec

        # current we only accept equal row and col sizes and strides
        if row_size != col_size:
            raise ValueError('Only square blocks are supported.')

        if row_stride != col_stride:
            raise ValueError('Only equal row/column stride is supported.')

        if row_size <= 0:
            raise ValueError('Block size must be positive.')

        if row_stride <= 0:
            raise ValueError('Block stride must be positive.')


@dataclasses.dataclass(frozen=True)
class PartitionResults:
    '''Container for partition results.'''
    partition_fpaths: geo_core.BlocksPartition
    raw_splits: split.SplitsResult
    hydration: split.HydrationResults


# -------------------------------Public Function-------------------------------
def create_blocks_partition(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_class_counts: dict[tuple[int, int], list[int]],
    valid_blocks: dict[tuple[int, int], str],
    config: PartitionParameters,
    *,
    ext_test_blks: list[str] | None
) -> PartitionResults:
    '''Split blocks with spatial safety and class balance.'''
    # ----- initial splitting from ratios
    raw_splits = split.stratified_splitter(
        base_class_counts,
        val_ratio=config.val_test_ratios[0],
        test_ratio=(0 if ext_test_blks else config.val_test_ratios[1]),
        weight_mode = 'inverse' # default
    )

    # ------ hydration process
    # filter candidate blocks for hydration
    safe_candidates = split.filter_safe_tiles(
        list(valid_class_counts.keys()),
        raw_splits.val + raw_splits.test,
        block_size=config.block_spec[0],
        block_stride=config.block_spec[2],
        buffer_steps=config.buffer_step
    )

    # score and rank the safe candidate tiles
    blocks_to_score = {
        k: v for k, v in valid_class_counts.items()
        if k in safe_candidates
    }
    ranked_candidates = split.score_blocks(
        list(raw_splits.global_class_count),
        blocks_to_score,
        reward=tuple(config.reward_ratios.keys()),
        alpha=config.scoring_alpha,
        beta=config.scoring_beta
    )

    # hydrate using the safe candidates
    hydration_results = split.hydrate_train_split(
        list(raw_splits.train_class_count),
        ranked_candidates,
        target_ratios=config.reward_ratios,
        max_skew_rate=config.max_skew_rate
    )

    # ----- final blocks partitions
    blocks_partition = _finalize_partition(
        valid_blocks,
        raw_splits,
        hydration_results.hydrated_train_blocks,
        ext_test_blks=ext_test_blks
    )

    return PartitionResults(
        partition_fpaths=blocks_partition,
        raw_splits=raw_splits,
        hydration=hydration_results
    )

# ----- internal helpers
def _finalize_partition(
    valid_blocks: dict[tuple[int, int], str],
    splits: split.SplitsResult,
    additional_train: list[tuple[int, int]],
    *,
    ext_test_blks: list[str] | None
) -> geo_core.BlocksPartition:
    '''Finalize the partition process with leakage sanity checks.'''

    def _index_fpath(fpaths: list[str]) -> dict[str, str]:
        '''Index block file paths by block name no file extension.'''
        indexed: dict[str, str] = {}
        for fpath in fpaths:
            filename = os.path.basename(fpath)
            name, _ = os.path.splitext(filename)
            indexed[name] = fpath # name is the same as core.xy_name()
        return indexed

    # get block fpaths for each split
    train = [valid_blocks[c] for c in splits.train + additional_train]
    val = [valid_blocks[c] for c in splits.val]
    test = [valid_blocks[c] for c in splits.test] + (ext_test_blks or [])

    # leakage sanity checks
    leak = set(train) & set(val)
    if leak:
        raise ValueError (f'Data leaked between train and val! {leak}')

    leak = set(train) & set(test)
    if leak:
        raise ValueError(f'Data leaked between train and test! {leak}')

    return {
        'train': _index_fpath(train),
        'val': _index_fpath(val),
        'test': _index_fpath(test)
    }
