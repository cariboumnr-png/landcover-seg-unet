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
import time
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform as transform
import landseg.geopipe.transform.common as common
import landseg.geopipe.transform.data_partition.split as split
import landseg.geopipe.transform.data_partition.stats as stats

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
LabelStatsCtrl = artifacts.Controller[dict[str, list[int]]]
ReportCtrl = artifacts.Controller[common.DataPartitionReport]

# -------------------------------Public Function-------------------------------
def run_datablocks_partition(
    parsed_catalog: transform.DataBlocksView,
    paths: artifacts.TransformPaths,
    partition_config: split.PartitionParameters,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: common.TransformLogger,
) -> None:
    '''
    Partition canonical data blocks into train/val/test splits.

    Loads the foundation catalog, optionally incorporates external test
    (holdout) blocks, and performs stratified splitting followed by
    spatially safe hydration of training blocks. Writes split manifests
    and label statistics for downstream normalization and schema generation.

    Args:
        parsed_catalog: DataBlocksView with loaded blocks.
        paths: Output paths container.
        partition_config: Parameters dict guiding split and hydration
            behavior.
        logger: Logger for progress and diagnostic output.
    '''
    start_time = time.perf_counter()

    # partition block filepaths JSON controller
    ctrl = PartitionCtrl(paths.splits_source_blocks, policy)
    partition_fpaths = ctrl.fetch()
    loaded = partition_fpaths is not None

    if not partition_fpaths:
        # get split blocks and persist the main results
        partition_results = split.create_blocks_partition(
            parsed_catalog.dev_base_class_counts,
            parsed_catalog.dev_valid_class_counts,
            parsed_catalog.dev_blocks,
            partition_config,
            ext_test_blks=parsed_catalog.external_test_blocks
        )
        partition_fpaths = partition_results.partition_fpaths
        ctrl.persist(partition_fpaths)
        logger.log('INFO', '[CHECKPOINT] Created dataset partition splits')

        duration = time.perf_counter() - start_time
        report = _report(
            partition_results,
            focal_head=parsed_catalog.focal_head,
            loaded=loaded,
            duration=duration
        )

    else:
        logger.log('INFO', '[CHECKPOINT] Loaded dataset partition splits')
        # partition report JSON controller
        duration = time.perf_counter() - start_time
        ctrl = ReportCtrl(paths.report, policy)
        report = ctrl.fetch()
        assert report # typing

    logger.set_data_partition_report(report)

    # label count results JSON controller
    ctrl = LabelStatsCtrl(paths.label_stats, policy)
    # iterate current traing blocks to get label class counts
    lbl_stats = stats.count_label(list(partition_fpaths['train'].values()))
    ctrl.persist(lbl_stats)


def _report(
    partition_results: split.PartitionResults,
    *,
    focal_head: str,
    loaded: bool,
    duration: float
) -> common.DataPartitionReport:
    '''Summarize class count & ratio changes.'''

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
        'status': 'loaded' if loaded else 'created',
        'duration_sec': duration,
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
