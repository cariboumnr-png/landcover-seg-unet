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

'''doc'''

# local imports
import landseg.geopipe.pipeline.transform.normal_blocks as normal_blocks
import landseg.utils as utils

def build_normalized_blocks(root_dir: str):
    '''doc'''

    # transform dir
    out_dir = f'{root_dir}/transform'

    # load source blocks file lists
    source: dict[str, list[str]] = utils.load_json(f'{out_dir}/sources.json')

    # get source by split
    train = set(source['train'])
    val = set(source['val'])
    test = set(source['test'])

    # aggregate stats on training blocks
    stats = normal_blocks.aggregate_image_stats(train)
    utils.write_json(f'{out_dir}/stats.json', stats)
    utils.hash_artifacts(f'{out_dir}/stats.json')

    # build normalized blocks for each split
    train_fpaths = normal_blocks.normalize_blocks(train, stats, f'{out_dir}/train_blocks')
    utils.write_json(f'{out_dir}/train_blocks.json', train_fpaths)
    utils.hash_artifacts(f'{out_dir}/train_blocks.json')

    val_fpaths = normal_blocks.normalize_blocks(val, stats, f'{out_dir}/val_blocks')
    utils.write_json(f'{out_dir}/val_blocks.json', val_fpaths)
    utils.hash_artifacts(f'{out_dir}/val_blocks.json')

    test_fpaths = normal_blocks.normalize_blocks(test, stats, f'{out_dir}/test_blocks')
    utils.write_json(f'{out_dir}/test_blocks.json', test_fpaths)
    utils.hash_artifacts(f'{out_dir}/test_blocks.json')
