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

    # partition results controller
    ctrl = PartitionCtrl(paths.splits_source_blocks, policy)
    splits_src = ctrl.fetch()
    loaded = splits_src is not None

    if not splits_src:
        # get split blocks and persist the main results
        splits_src, summary = split.create_blocks_partition(
            parsed_catalog.dev_base_class_counts,
            parsed_catalog.dev_valid_class_counts,
            parsed_catalog.dev_blocks,
            partition_config,
            ext_test_blks=parsed_catalog.external_test_blocks
        )
        ctrl.persist(splits_src)
        # if split is run persist a splits summary as well
        artifacts.Controller(paths.splits_summary, policy).persist(summary)
        logger.log('INFO', '[CHECKPOINT] Created dataset partition splits')
        # log partition summary
        for m in summary.items():
            logger.log('INFO', f'{m[0]}: {m[1]}')
    else:
        logger.log('INFO', '[CHECKPOINT] Loaded dataset partition splits')

    # label count results controller
    ctrl = LabelStatsCtrl(paths.label_stats, policy)
    lbl_stats = ctrl.fetch()

    if not lbl_stats:
        # iterate current traing blocks to get label class counts
        lbl_stats = stats.count_label(list(splits_src['train'].values()))
        ctrl.persist(lbl_stats)

    # compile partition report
    duration = time.perf_counter() - start_time
    summary_ctrl = artifacts.Controller(paths.splits_summary, policy)
    summary_data = summary_ctrl.fetch() or {}

    report: common.DataPartitionReport = {
        'status': 'loaded' if loaded else 'created',
        'duration_sec': duration,
        'training_blocks': len(splits_src['train']),
        'validation_blocks': len(splits_src['val']),
        'test_blocks': len(splits_src['test']),
        'base_label_count': summary_data.get('base_label_count', []),
        'hydrated_label_count': summary_data.get('hydrated_label_count', []),
    }
    logger.set_data_partition_report(report)
