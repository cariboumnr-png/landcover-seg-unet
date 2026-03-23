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
import landseg.grid_generator as grid
import landseg._ingest_dataset as dataset
import landseg.utils as utils

def dev_test(config: configs.RootConfig):
    '''Test..'''

    logger = utils.Logger('test', './test.log')

    # config overrides
    config.prep.grid.id = 'grid_row_256_128_col_256_128'
    config.prep.grid.tile_overlap.row = 128
    config.prep.grid.tile_overlap.col = 128

    # prep world grid
    g = grid.prep_world_grid(config.inputs.extent, config.prep.grid, logger)

    # dataset pipeline test
    dataset.test_pipeline(g, config, logger)
