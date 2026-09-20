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
Dataset partitioning pipeline.

Consumes a canonical blocks catalog and produces experiment-specific
dataset splits (train/val/test) based on stratified sampling, spatial
buffering, and class-balance heuristics. Outputs split manifests and
label statistics for downstream normalization and schema generation.

Public APIs:
    - run_datablocks_partition: partition blocks into train/val/test.
'''

# standard imports
from __future__ import annotations
import time
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.preparation as contracts
import landseg.geopipe.prepare as prepare
import landseg.geopipe.prepare.dataset as dataset
import landseg.geopipe.prepare.partition.orchestration as orchestration


# ----- typing aliases
PartitionCtrl = artifacts.Controller[contracts.BlocksPartition]
SplitsSummaryCtrl = artifacts.Controller[contracts.PartitionSummary]


# ----- private types
class _PipelinePaths(typing.Protocol):
    @property
    def splits_source_blocks(self) -> str: ...
    @property
    def splits_summary(self) -> str: ...


# ----- public functions
def run_datablocks_partition(
    context: dataset.DatasetView,
    paths: _PipelinePaths,
    partition_config: orchestration.PartitionParameters,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: prepare.PreparationLogger,
) -> None:
    '''
    Partition canonical data blocks into train/val/test splits.

    Consumes the dataset preparation view, performs stratified
    splitting followed by spatially safe hydration of training blocks,
    and writes split manifests and label statistics for downstream
    normalization and schema generation.

    Args:
        context:
            DatasetView with loaded catalog and resolved semantics.
        paths:
            pipeline artifact output paths container.
        partition_config:
            parameters dict guiding split and hydration behavior.
        policy:
            artifact lifecycle policy guiding rebuild behavior.
        logger:
            logger for progress and diagnostic output.
    '''
    start_time = time.perf_counter()

    # ensure canvas CRS and transform default from context
    if partition_config.canvas_crs == 'EPSG:3161' and context.crs:
        partition_config.canvas_crs = context.crs
    if partition_config.canvas_transform is None and context.transform:
        partition_config.canvas_transform = context.transform

    # partition fpaths and summary JSON controller
    partition_ctrl = PartitionCtrl(paths.splits_source_blocks, policy)
    partition_fpaths = partition_ctrl.fetch()
    summary_ctrl = SplitsSummaryCtrl(paths.splits_summary, policy)
    summary = summary_ctrl.fetch()

    if (
        policy == artifacts.LifecyclePolicy.REBUILD
        or not (partition_fpaths and summary)
    ):

        # blocks fpaths
        partition_results = orchestration.create_blocks_partition(
            context.base_class_counts,
            context.valid_class_counts,
            context.valid_blocks,
            partition_config,
            ext_test_blks=context.external_test_blocks,
            logger=logger,
        )

        partition_fpaths = partition_results.partition_fpaths
        partition_ctrl.persist(partition_fpaths)

        # summary
        splits_summary = _build_splits_summary(
            partition_results,
            focal_head=context.focal_head,
        )
        summary_ctrl.persist(splits_summary)

        status = 'created'
        logger.log('INFO', '[CHECKPOINT] Created dataset partition splits')
    else:
        status = 'loaded'
        logger.log('INFO', '[CHECKPOINT] Loaded dataset partition splits')

    duration = time.perf_counter() - start_time
    report: contracts.DataPartitionReport = {
        'status': status,
        'duration_sec': duration
    }
    logger.set_data_partition_report(report)


# ----- private helpers
def _build_splits_summary(
    partition_results: orchestration.PartitionResults,
    *,
    focal_head: str,
) -> contracts.PartitionSummary:
    '''Summarize class count and distribution changes across splits.'''
    splits = partition_results.raw_splits
    start_count = list(splits.global_class_count)
    distb = splits.class_distributions

    current_n_train = len(partition_results.partition_fpaths['train'])
    n_train_diff = current_n_train - len(splits.train)

    if n_train_diff: # hydration performed
        current_count = partition_results.hydration.hydrated_class_count

        count_diff = [x - y for (x, y) in zip(current_count, start_count)]

        start_sum = sum(start_count) or 1.0
        start_per = [float(x / start_sum) for x in start_count]

        current_sum = sum(current_count) or 1.0
        current_per = [float(x / current_sum) for x in current_count]

        per_diff = [float(x - y) for (x, y) in zip(current_per, start_per)]

    else:
        current_count = []
        count_diff = []
        current_per = []
        per_diff = []

    return {
        'original_splits': {
            'training': {
                'num_of_blocks': len(splits.train),
                'class_count': splits.class_counts['train'],
                'class_distribution': distb['train']
            },
            'validation': {
                'num_of_blocks': len(splits.val),
                'class_count': splits.class_counts['val'],
                'class_distribution': distb['val']
            },
            'testing': {
                'num_of_blocks': len(splits.test),
                'class_count': splits.class_counts['test'],
                'class_distribution': distb['test']
            },
        },
        'hydration': {
            'performed': bool(n_train_diff),
            'focal_head': focal_head,
            'stop_reason': partition_results.hydration.info,
            'n_training_blocks': current_n_train,
            'n_training_blocks_change': n_train_diff,
            'hydrated_class_count': current_count,
            'class_count_change': count_diff,
            'hydrated_class_distribution': current_per,
            'class_distribution_change': per_diff
        }
    }
