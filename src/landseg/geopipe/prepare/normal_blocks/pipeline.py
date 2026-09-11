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

# pylint: disable=missing-function-docstring

'''
Image normalization and block materialization pipeline.

Consumes raw block split manifests, aggregates image statistics from
training blocks only, normalizes all splits using these statistics, and
writes normalized block artifacts along with updated split mappings.
'''

# standard imports
import time
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.common as common
import landseg.geopipe.prepare.normal_blocks.normalize as normalize
import landseg.geopipe.prepare.normal_blocks.stats as stats

# --------------------------------private types--------------------------------
class _PipelinePaths(typing.Protocol):
    '''Typed pipeline-specific paths container.'''
    @property
    def splits_source_blocks(self) -> str: ...
    @property
    def image_stats(self) -> str: ...
    @property
    def splits_transformed_blocks(self) -> str: ...
    @property
    def train_blocks(self) -> str: ...
    @property
    def val_blocks(self) -> str: ...
    @property
    def test_blocks(self) -> str: ...

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
ImageStatsCtrl = artifacts.Controller[dict[str, geo_core.ImageBandStats]]


# ----- `run_normalize_blocks` execution
def run_normalize_blocks(
    paths: _PipelinePaths,
    *,
    channel_indices: list[int] | None = None,
    target_reclass: dict[str, geo_core.LabelScheme | None] | None = None,
    policy: artifacts.LifecyclePolicy,
    logger: common.PreparationLogger
):
    '''
    Build normalized data blocks from raw block splits.

    Loads the raw `block_source.json`, computes per-band image statistics
    using training blocks only, and applies normalization consistently to
    train, validation, and test splits. Normalized blocks are written to
    split-specific directories and registered in `block_splits.json`.

    Args:
        paths: Transform paths container.
        channel_indices: Optional list of 0-based channel indices to
            select for the normalized blocks.
        target_reclass: Optional dictionary of target reclassification
            settings per label layer.
        policy: Lifecycle policy guiding rebuild behavior.
        logger: Logger for progress and diagnostic output.
    '''
    start_time = time.perf_counter()

    # load source blocks file lists
    ctrl = PartitionCtrl.load_json_or_fail(paths.splits_source_blocks)
    src = ctrl.fetch()

    # get source by split
    train = set(src['train'].values())
    val = set(src['val'].values())
    test = set(src['test'].values())

    # aggregate stats on training blocks
    ctrl = ImageStatsCtrl(paths.image_stats, policy)
    aggregated_stats = ctrl.fetch()
    if policy != artifacts.LifecyclePolicy.REBUILD and aggregated_stats:
        logger.log('INFO', '[CHECKPOINT] Loaded image stats from training split')
    else:
        aggregated_stats = stats.aggregate_image_stats(
            train, channel_indices=channel_indices
        )
        ctrl.persist(aggregated_stats)
        logger.log('INFO', '[CHECKPOINT] Created image stats from training split')

    # load or build normalized blocks for each split
    ctrl = PartitionCtrl(paths.splits_transformed_blocks, policy)
    transform = ctrl.fetch()
    loaded = transform is not None

    purged_total = 0
    if policy != artifacts.LifecyclePolicy.REBUILD and transform:
        logger.log('INFO', '[CHECKPOINT] Loaded normalized dataset blocks')
    else:
        transform, purged_total = _normalize(
            (train, val, test),
            aggregated_stats,
            paths,
            channel_indices=channel_indices,
            target_reclass=target_reclass,
            logger=logger
        )
        ctrl.persist(transform)
        logger.log('INFO', '[CHECKPOINT] Created normalized dataset blocks')

    # compile report
    duration = time.perf_counter() - start_time
    report: common.NormalizationReport = {
        'status': 'loaded' if loaded else 'created',
        'duration_sec': duration,
        'unwanted_blocks_removed': purged_total,
        'rebuild': False,
        'stats_filepath': paths.image_stats,
    }
    logger.set_normalization_report(report)


# ----- `_normalize` helper
def _normalize(
    splits: tuple[set[str], set[str], set[str]],
    aggregated_stats: dict[str, geo_core.ImageBandStats],
    paths: _PipelinePaths,
    *,
    channel_indices: list[int] | None = None,
    target_reclass: dict[str, geo_core.LabelScheme | None] | None = None,
    logger: common.PreparationLogger
):
    '''Normalize each split.'''
    train_split, val_split, test_split = splits

    purged_total = 0
    train_norm, purged = normalize.normalize_blocks(
        train_split,
        aggregated_stats,
        paths.train_blocks,
        channel_indices=channel_indices,
        target_reclass=target_reclass,
    )
    if purged:
        purged_total += purged
        logger.log('DEBUG', f'{purged} stale training block files removed')
    val_norm, purged = normalize.normalize_blocks(
        val_split,
        aggregated_stats,
        paths.val_blocks,
        channel_indices=channel_indices,
        target_reclass=target_reclass,
    )
    if purged:
        purged_total += purged
        logger.log('DEBUG', f'{purged} stale validation block files removed')
    test_norm, purged = normalize.normalize_blocks(
        test_split,
        aggregated_stats,
        paths.test_blocks,
        channel_indices=channel_indices,
        target_reclass=target_reclass,
    )
    if purged:
        purged_total += purged
        logger.log('DEBUG', f'{purged} stale testing block files removed')

    transform = {
        'train': train_norm,
        'val': val_norm,
        'test': test_norm
    }

    return transform, purged_total
