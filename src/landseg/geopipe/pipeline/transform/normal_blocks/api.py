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
    train = utils.load_json(f'{out_dir}/train_blocks.json')
    val = utils.load_json(f'{out_dir}/val_blocks.json')
    test = utils.load_json(f'{out_dir}/test_blocks.json')

    # aggregate stats on training blocks
    stats = normal_blocks.aggregate_image_stats(train)

    # build normalized blocks for each split
    normal_blocks.normalize_blocks(train, stats, f'{out_dir}/train_blocks')
    normal_blocks.normalize_blocks(val, stats, f'{out_dir}/val_blocks')
    normal_blocks.normalize_blocks(test, stats, f'{out_dir}/test_blocks')
