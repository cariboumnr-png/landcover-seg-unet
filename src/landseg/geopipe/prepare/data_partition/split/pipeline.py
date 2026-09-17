# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Core partitioning pipeline for data blocks.

Coordinates stratified sampling, AOI spatial filtering, buffer safety
checks, and candidate block hydration to produce train, validation,
and testing splits without data leakage.

Public APIs:
    - PartitionParameters: configuration for data block partitioning.
    - PartitionResults: container for partitioned splits and hydration.
    - create_blocks_partition: split blocks safely without leakage.
'''

# standard imports
import dataclasses
import os
# third-party imports
import rasterio
import rasterio.transform
# local imports
import landseg.geopipe.prepare.common as common
import landseg.geopipe.prepare.data_partition.split as split


# ----- public dataclasses
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
    train_aoi: str | None = None
    val_aoi: str | None = None
    test_aoi: str | None = None
    aoi_min_overlap: float = 0.5
    canvas_crs: str = 'EPSG:3161'
    canvas_transform: rasterio.transform.Affine | None = None


@dataclasses.dataclass(frozen=True)
class PartitionResults:
    '''Container for partitioned splits and hydration results.'''
    partition_fpaths: common.BlocksPartition
    raw_splits: split.SplitsResult
    hydration: split.HydrationResults


# ----- public functions
def create_blocks_partition(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_class_counts: dict[tuple[int, int], list[int]],
    valid_blocks: dict[tuple[int, int], str],
    config: PartitionParameters,
    *,
    ext_test_blks: list[str] | None = None,
    logger: common.PreparationLogger | None = None,
) -> PartitionResults:
    '''
    Split blocks with spatial safety, AOI selection, and class balance.

    Performs stratified partitioning or AOI filtering, computes buffer
    zones to eliminate spatial leakage, and optionally hydrates the
    training set with oversampled candidate blocks.

    Args:
        base_class_counts:
            mapping of base block coordinates to focal head counts.
        valid_class_counts:
            mapping of all valid block coordinates to focal head counts.
        valid_blocks:
            mapping of valid block coordinates to artifact file paths.
        config:
            partitioning parameters and buffer configuration.
        ext_test_blks:
            optional external holdout test block file paths.
        logger:
            optional logger for progress and diagnostic messages.

    Returns:
        PartitionResults:
            split block paths, raw split coordinates, and hydration
            diagnostics.
    '''
    has_aoi = bool(config.train_aoi or config.val_aoi or config.test_aoi)

    if has_aoi:
        raw_splits = _split_by_aoi(
            base_class_counts,
            valid_blocks,
            config,
            ext_test_blks=ext_test_blks,
            logger=logger,
        )
    else:
        raw_splits = split.stratified_splitter(
            base_class_counts,
            val_ratio=config.val_test_ratios[0],
            test_ratio=(0.0 if ext_test_blks else config.val_test_ratios[1]),
            weight_mode='inverse',
        )

    # hydration process (optional)
    if bool(config.reward_ratios):
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
    else:
        hydration_results = split.HydrationResults()

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


# ----- private helpers
def _split_by_aoi(
    base_class_counts: dict[tuple[int, int], list[int]],
    valid_blocks: dict[tuple[int, int], str],
    config: PartitionParameters,
    *,
    ext_test_blks: list[str] | None,
    logger: common.PreparationLogger | None,
) -> split.SplitsResult:
    '''Resolve AOI partitions and split remaining blocks.'''
    transform = config.canvas_transform or rasterio.transform.Affine.identity()
    block_size = (config.block_spec[0], config.block_spec[1])

    aoi_res = split.resolve_aoi_partitions(
        list(valid_blocks.keys()),
        train_aoi=config.train_aoi,
        val_aoi=config.val_aoi,
        test_aoi=config.test_aoi,
        block_size=block_size,
        canvas_crs=config.canvas_crs,
        canvas_transform=transform,
        min_overlap=config.aoi_min_overlap,
        logger=logger,
    )

    test_coords = [c for c in aoi_res.test if c in base_class_counts]
    val_coords = [c for c in aoi_res.val if c in base_class_counts]
    train_coords = [c for c in aoi_res.train if c in base_class_counts]
    unassigned = [c for c in aoi_res.unassigned if c in base_class_counts]

    # auto-split unassigned blocks if ratio requested
    if unassigned:
        unassigned_counts = {c: base_class_counts[c] for c in unassigned}
        auto_test_ratio = (
            0.0
            if (config.test_aoi or ext_test_blks)
            else config.val_test_ratios[1]
        )
        auto_val_ratio = 0.0 if config.val_aoi else config.val_test_ratios[0]

        if auto_val_ratio > 0.0 or auto_test_ratio > 0.0:
            auto_splits = split.stratified_splitter(
                unassigned_counts,
                val_ratio=auto_val_ratio,
                test_ratio=auto_test_ratio,
                weight_mode='inverse',
            )
            val_coords.extend(auto_splits.val)
            test_coords.extend(auto_splits.test)
            if not config.train_aoi:
                train_coords.extend(auto_splits.train)
        elif not config.train_aoi:
            train_coords.extend(unassigned)

    # enforce spatial buffer on training blocks against val and test
    if config.buffer_step > 0 and (val_coords or test_coords):
        safe_train = split.filter_safe_tiles(
            train_coords,
            val_coords + test_coords,
            block_size=config.block_spec[0],
            block_stride=config.block_spec[2],
            buffer_steps=config.buffer_step,
        )
        if len(safe_train) < len(train_coords) and logger is not None:
            excluded = len(train_coords) - len(safe_train)
            logger.log(
                'WARNING',
                f'Pruned {excluded} training block(s) bordering buffer zone.'
            )
        train_coords = safe_train

    # aggregate class counts
    n_classes = (
        len(next(iter(base_class_counts.values())))
        if base_class_counts
        else 0
    )
    train_cls = [0] * n_classes
    val_cls = [0] * n_classes
    test_cls = [0] * n_classes
    global_cls = [0] * n_classes

    for c in train_coords:
        for idx, count in enumerate(base_class_counts[c]):
            train_cls[idx] += count
            global_cls[idx] += count

    for c in val_coords:
        for idx, count in enumerate(base_class_counts[c]):
            val_cls[idx] += count
            global_cls[idx] += count

    for c in test_coords:
        for idx, count in enumerate(base_class_counts[c]):
            test_cls[idx] += count
            global_cls[idx] += count

    return split.SplitsResult(
        train=train_coords,
        val=val_coords,
        test=test_coords,
        train_class_count=tuple(train_cls),
        val_class_count=tuple(val_cls),
        test_class_count=tuple(test_cls),
        global_class_count=tuple(global_cls),
    )


def _finalize_partition(
    valid_blocks: dict[tuple[int, int], str],
    splits: split.SplitsResult,
    additional_train: list[tuple[int, int]],
    *,
    ext_test_blks: list[str] | None,
) -> common.BlocksPartition:
    '''Finalize the partition process with leakage sanity checks.'''
    def _index_fpath(fpaths: list[str]) -> dict[str, str]:
        '''Index block file paths by block name without extension.'''
        indexed: dict[str, str] = {}
        for fpath in fpaths:
            filename = os.path.basename(fpath)
            name, _ = os.path.splitext(filename)
            indexed[name] = fpath # name is the same as core.xy_name()
        return indexed

    train = [
        valid_blocks[c]
        for c in splits.train + additional_train
        if c in valid_blocks
    ]
    val = [valid_blocks[c] for c in splits.val if c in valid_blocks]
    test = (
        [valid_blocks[c] for c in splits.test if c in valid_blocks]
        + (ext_test_blks or [])
    )

    # leakage sanity checks
    leak = set(train) & set(val)
    if leak:
        raise ValueError(f'Data leaked between [train] and [val]! {leak}')

    leak = set(train) & set(test)
    if leak:
        raise ValueError(f'Data leaked between [train] and [test]! {leak}')

    leak = set(val) & set(test)
    if leak:
        raise ValueError(f'Data leaked between [val] and [test]! {leak}')

    return {
        'train': _index_fpath(train),
        'val': _index_fpath(val),
        'test': _index_fpath(test)
    }
