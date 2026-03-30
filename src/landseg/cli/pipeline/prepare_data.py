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
Dev Test Ground
'''

# local imports
import landseg.configs as configs
import landseg.geopipe.transform as transform
import landseg.utils as utils

def prepare(config: configs.RootConfig):
    '''Train data preparation pipeline.'''

    logger = utils.Logger('test', './test.log')
    artifacts_root = './experiment/artifacts/'

    # datablocks partition
    partition_cfg = transform.PartitionConfig(
        val_test_ratios=(0.1, 0),
        buffer_step=1,
        reward_ratios={2: 5.0, 4: 5.0},
        scoring_alpha=1.0,
        scoring_beta=config.prep.data.scoring.beta,
        max_skew_rate=10.0,
        block_spec=(256, 128, 256, 128)
    )
    transform.partition_blocks(artifacts_root, partition_cfg, logger)

    # normalize
    transform.build_normalized_blocks(artifacts_root)

    # build schema
    transform.build_schema_full(f'{artifacts_root}/transform')
