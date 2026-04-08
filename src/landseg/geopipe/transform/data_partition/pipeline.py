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

# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform as transform
import landseg.geopipe.transform.data_partition as data_partition
import landseg.utils as utils

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
LabelStatsCtrl = artifacts.Controller[dict[str, list[int]]]

# -------------------------------Public Function-------------------------------
def run_datablocks_partition(
    logger: utils.Logger,
    paths: artifacts.TransformPaths,
    parsed_catalog: transform.DataBlocksView,
    partition_config: data_partition.PartitionParameters,
    *,
    policy: artifacts.LifecyclePolicy
) -> None:
    '''
    Partition canonical data blocks into train/val/test splits.

    Loads the foundation catalog, optionally incorporates external test
    (holdout) blocks, and performs stratified splitting followed by
    spatially safe hydration of training blocks. Writes split manifests
    and aggregated label statistics to the transform directory.

    Artifacts written:
    - `block_source.json`: block file paths indexed by split
    - `label_stats.json`: aggregated label counts over training blocks

    Args:
        foundation_root: Root directory containing canonical catalog
            artifacts.
        transform_root: Output directory for transform-stage artifacts.
        partition_config: Parameters controlling split and balancing
            behavior.
        logger: Logger for progress and diagnostic output.
    '''

    # get a child logger
    logger = logger.get_child('split')

    # partition results controller
    ctrl = PartitionCtrl(paths.splits_source_blocks, policy)
    splits_src = ctrl.fetch()

    if not splits_src:
        # get split blocks and persist the main results
        logger.log('INFO', 'Split data blocks')
        splits_src, summary = data_partition.create_blocks_partition(
            parsed_catalog.dev_base_class_counts,
            parsed_catalog.dev_valid_class_counts,
            parsed_catalog.dev_blocks,
            partition_config,
            ext_test_blks=parsed_catalog.external_test_blocks
        )
        ctrl.persist(splits_src)
        # if split is run persist a splits summary as well
        artifacts.Controller(paths.splits_summary, policy).persist(summary)
        # log partition summary
        for m in summary.items():
            logger.log('INFO', f'{m[0]}: {m[1]}')

    # label count results controller
    ctrl = LabelStatsCtrl(paths.label_stats, policy)
    lbl_stats = ctrl.fetch()

    if not lbl_stats:
        # iterate current traing blocks to get label class counts
        lbl_stats = _count_label(list(splits_src['train'].values()))
        ctrl.persist(lbl_stats)

def _count_label(block_file_list: list[str]) -> dict[str, list[int]]:
    '''Aggregate label class counts across a list of block files.'''

    # iterate current traing blocks to get label class counts
    lbl_stats: dict[str, list[int]] = {}
    for fpath in block_file_list:
        blk_meta = geo_core.DataBlock.load(fpath).meta
        for channel, counts in blk_meta['label_count'].items():
            cls_count = numpy.asarray(counts)
            if channel in lbl_stats:
                lbl_stats[channel] += cls_count
            else:
                lbl_stats[channel] = list(cls_count)
            lbl_stats[channel] = [int(x) for x in lbl_stats[channel]]
    return lbl_stats
