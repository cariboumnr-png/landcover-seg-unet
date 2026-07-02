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

# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.normal_blocks.normalize as normalize
import landseg.geopipe.transform.normal_blocks.stats as stats

# typing aliases
PartitionCtrl = artifacts.Controller[geo_core.BlocksPartition]
ImageStatsCtrl = artifacts.Controller[dict[str, geo_core.ImageBandStats]]

def run_normalize_blocks(
    paths: artifacts.TransformPaths,
    *,
    policy: artifacts.LifecyclePolicy
):
    '''
    Build normalized data blocks from raw block splits.

    Loads the raw `block_source.json`, computes per-band image statistics
    using training blocks only, and applies normalization consistently to
    train, validation, and test splits. Normalized blocks are written to
    split-specific directories and registered in `block_splits.json`.

    Artifacts written:
    - `image_stats.json`: train-only image statistics
    - `block_splits.json`: normalized block paths indexed by split

    Args:
        root_dir: Transform root directory containing split manifests and
            receiving normalized artifacts.
    '''

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


    # build normalized blocks for each split
    ctrl = PartitionCtrl(paths.splits_transformed_blocks, policy)
    transform = ctrl.fetch()
    if not transform:
        transform = {
            'train': normalize.normalize_blocks(train, agg_stats, train_dpath),
            'val': normalize.normalize_blocks(val, agg_stats, val_dpath),
            'test': normalize.normalize_blocks(test, agg_stats, test_dpath)
        }
        ctrl.persist(transform)
