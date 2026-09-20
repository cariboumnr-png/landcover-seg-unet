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

Public APIs:
    - run_materialize_blocks: orchestrate stats aggregation and block
      materialization.
'''

# standard imports
import time
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.prepare as prepare
import landseg.geopipe.prepare.dataset as dataset
import landseg.geopipe.prepare.materialize.materialize as materialize
import landseg.geopipe.prepare.materialize.schema as schema
import landseg.geopipe.prepare.materialize.stats as stats


# ----- typing aliases
PartitionCtrl = artifacts.Controller[contracts.BlocksPartition]
LabelStatsCtrl = artifacts.Controller[dict[str, list[int]]]
ImageStatsCtrl = artifacts.Controller[dict[str, contracts.ImageBandStats]]


# ----- private types
class _PipelinePaths(typing.Protocol):
    @property
    def splits_source_blocks(self) -> str: ...
    @property
    def image_stats(self) -> str: ...
    @property
    def label_stats(self) -> str: ...
    @property
    def splits_prepared_blocks(self) -> str: ...
    @property
    def train_blocks(self) -> str: ...
    @property
    def val_blocks(self) -> str: ...
    @property
    def test_blocks(self) -> str: ...
    @property
    def schema(self) -> str: ...


# ----- public functions
def run_materialize_blocks(
    paths: _PipelinePaths,
    context: dataset.DatasetView,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare.PreparationLogger,
) -> None:
    '''
    Build normalized data blocks from raw block splits.

    Loads raw block split manifests, computes per-band image statistics
    using training blocks only, aggregates label class counts, and
    materializes normalized blocks for all splits.

    Args:
        paths:
            pipeline artifact and directory paths container.
        context:
            dataset view containing features and target hierarchy.
        policy:
            lifecycle policy guiding rebuild behavior.
        logger:
            logger for progress and diagnostic output.
    '''
    start_time = time.perf_counter()

    # load source blocks file lists
    ctrl = PartitionCtrl.load_json_or_fail(paths.splits_source_blocks)
    src = ctrl.fetch()

    # get source by split
    train = set(src['train'].values())
    val = set(src['val'].values())
    test = set(src['test'].values())

    # label stats on training blocks - parse from context
    ctrl = LabelStatsCtrl(paths.label_stats, policy)
    lbl_counts = stats.count_label(
        context.catalog.raw_class_counts,
        list(src['train'].keys())
    )
    ctrl.persist(lbl_counts)

    # aggregate stats on training blocks
    ctrl = ImageStatsCtrl(paths.image_stats, policy)
    agg_stats = ctrl.fetch()
    if policy != artifacts.LifecyclePolicy.REBUILD and agg_stats:
        logger.log(
            'INFO', '[CHECKPOINT] Loaded image stats from training split'
        )
    else:
        agg_stats = stats.aggregate_image_stats(
            set(src['train'].values()),
            list(context.features.indices)
        )
        ctrl.persist(agg_stats)
        logger.log(
            'INFO', '[CHECKPOINT] Created image stats from training split'
        )

    # load or build normalized blocks for each split
    ctrl = PartitionCtrl(paths.splits_prepared_blocks, policy)
    prepared = ctrl.fetch()
    loaded = prepared is not None

    purged_total = 0
    if policy != artifacts.LifecyclePolicy.REBUILD and prepared:
        logger.log('INFO', '[CHECKPOINT] Loaded normalized dataset blocks')
    else:
        prepared, purged_total = _materialize(
            (train, val, test),
            agg_stats,
            context,
            paths,
            logger=logger,
        )
        ctrl.persist(prepared)
        logger.log('INFO', '[CHECKPOINT] Created normalized dataset blocks')

    # build schema
    logger.log('INFO', '[START] Prepared schema building')
    schema.build_schema(
        paths,
        context,
        policy=policy,
        logger=logger,
    )

    # compile report
    duration = time.perf_counter() - start_time
    report: contracts.NormalizationReport = {
        'status': 'loaded' if loaded else 'created',
        'duration_sec': duration,
        'unwanted_blocks_removed': purged_total,
        'rebuild': False,
        'stats_filepath': paths.image_stats,
    }
    logger.set_normalization_report(report)


# ----- private helpers
def _materialize(
    splits: tuple[set[str], set[str], set[str]],
    aggregated_stats: dict[str, contracts.ImageBandStats],
    context: dataset.DatasetView,
    paths: _PipelinePaths,
    *,
    logger: prepare.PreparationLogger,
) -> tuple[dict[str, dict[str, str]], int]:
    '''Materialize and normalize train, validation, and test splits.'''
    train_split, val_split, test_split = splits

    purged_total = 0
    train_norm, purged = materialize.materialize_blocks(
        train_split,
        aggregated_stats,
        context,
        paths.train_blocks,
    )
    if purged:
        purged_total += purged
        logger.log('DEBUG', f'{purged} stale training block files removed')
    val_norm, purged = materialize.materialize_blocks(
        val_split,
        aggregated_stats,
        context,
        paths.val_blocks,
    )
    if purged:
        purged_total += purged
        logger.log('DEBUG', f'{purged} stale validation block files removed')
    test_norm, purged = materialize.materialize_blocks(
        test_split,
        aggregated_stats,
        context,
        paths.test_blocks,
    )
    if purged:
        purged_total += purged
        logger.log('DEBUG', f'{purged} stale testing block files removed')

    prepared = {
        'train': train_norm,
        'val': val_norm,
        'test': test_norm
    }

    return prepared, purged_total
