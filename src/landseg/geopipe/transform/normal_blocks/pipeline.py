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

# pylint: disable=missing-function-docstring

'''
Image normalization and block materialization pipeline.

Consumes raw block split manifests, aggregates image statistics from
training blocks only, normalizes all splits using these statistics, and
writes normalized block artifacts along with updated split mappings.
'''

# standard imports
import os
import time
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.common as common
import landseg.geopipe.transform.normal_blocks.normalize as normalize
import landseg.geopipe.transform.normal_blocks.stats as stats

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
ImageStatsCtrl = artifacts.Controller[dict[str, geo_core.ImageBandStats]]

def run_normalize_blocks(
    paths: artifacts.TransformPaths,
    *,
    policy: artifacts.LifecyclePolicy,
    logger: common.TransformLogger
):
    '''
    Build normalized data blocks from raw block splits.

    Loads the raw `block_source.json`, computes per-band image statistics
    using training blocks only, and applies normalization consistently to
    train, validation, and test splits. Normalized blocks are written to
    split-specific directories and registered in `block_splits.json`.

    Args:
        paths: Transform paths container.
        policy: Lifecycle policy guiding rebuild behavior.
        logger: Logger for progress and diagnostic output.
    '''

    start_time = time.perf_counter()
    child_logger = logger.get_child('norm')

    # load source blocks file lists
    ctrl = PartitionCtrl.load_json_or_fail(paths.splits_source_blocks)
    src = ctrl.fetch()
    assert src # typing assertion

    # get source by split
    train = set(src['train'].values())
    val = set(src['val'].values())
    test = set(src['test'].values())

    # aggregate stats on training blocks
    ctrl = ImageStatsCtrl(paths.image_stats, policy)
    agg_stats = ctrl.fetch()
    if not agg_stats:
        agg_stats = stats.aggregate_image_stats(train)
        ctrl.persist(agg_stats)

    # save dirs
    train_dpath = paths.train_blocks
    val_dpath = paths.val_blocks
    test_dpath = paths.test_blocks

    # Pre-calculate purged blocks for the report
    purged_total = 0
    for dpath, block_set in [(train_dpath, train), (val_dpath, val), (test_dpath, test)]:
        if os.path.exists(dpath):
            names = {os.path.basename(b) for b in block_set}
            for fpath in os.listdir(dpath):
                if fpath.endswith('.npz') and fpath not in names:
                    purged_total += 1

    # build normalized blocks for each split
    ctrl = PartitionCtrl(paths.splits_transformed_blocks, policy)
    transform = ctrl.fetch()
    loaded = transform is not None
    if not transform:
        transform = {
            'train': normalize.normalize_blocks(train, agg_stats, train_dpath, logger=child_logger),
            'val': normalize.normalize_blocks(val, agg_stats, val_dpath, logger=child_logger),
            'test': normalize.normalize_blocks(test, agg_stats, test_dpath, logger=child_logger)
        }
        ctrl.persist(transform)

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
