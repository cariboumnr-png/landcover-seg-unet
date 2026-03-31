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
Image normalization and block materialization pipeline.

Consumes raw block split manifests, aggregates image statistics from
training blocks only, normalizes all splits using these statistics, and
writes normalized block artifacts along with updated split mappings.
'''

# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.normal_blocks as normal_blocks
import landseg.utils as utils

def run_normaliza_blocks(root_dir: str):
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
    src: geo_core.BlocksPartition = utils.load_json(f'{root_dir}/block_source.json')

    # get source by split
    train = set(src['train'].values())
    val = set(src['val'].values())
    test = set(src['test'].values())

    # aggregate stats on training blocks
    stats = normal_blocks.aggregate_image_stats(train)
    utils.write_json(f'{root_dir}/image_stats.json', stats)
    utils.hash_artifacts(f'{root_dir}/image_stats.json')

    # save dirs
    train_dpath = f'{root_dir}/train_blocks'
    val_dpath = f'{root_dir}/val_blocks'
    test_dpath = f'{root_dir}/test_blocks'

    # build normalized blocks for each split
    transform: geo_core.BlocksPartition = {
        'train': normal_blocks.normalize_blocks(train, stats,train_dpath),
        'val': normal_blocks.normalize_blocks(val, stats, val_dpath),
        'test': normal_blocks.normalize_blocks(test, stats, test_dpath)
    }

    # save and hash artifacts
    utils.write_json(f'{root_dir}/block_splits.json', transform)
    utils.hash_artifacts(f'{root_dir}/block_splits.json')
