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
import landseg.ingest_catalog as catalogue
import landseg.utils as utils

def dev_test(config: configs.RootConfig):
    '''Test..'''

    logger = utils.Logger('test', './test.log')

    # load/create world grid
    config.prep.grid.id = 'grid_row_256_128_col_256_128'
    config.prep.grid.tile_overlap.row = 128
    config.prep.grid.tile_overlap.col = 128
    g1 = grid.prep_world_grid(config.inputs.extent, config.prep.grid, logger)
    config.prep.grid.id = 'grid_row_256_0_col_256_0'
    config.prep.grid.tile_overlap.row = 0
    config.prep.grid.tile_overlap.col = 0
    g2 = grid.prep_world_grid(config.inputs.extent, config.prep.grid, logger)

    catalogue.catalogue_pipeline_test(
        [g1, g2],
        config,
        logger,
        build_a_block=False
    )

SAMPLE_CATALOG_EACH_BLOCK = {
    "block_name": "row_000001_col_000001",
    "file_path": "../blocks/row_000001_col_000001.npz",
    "coordinates": [1, 1],
    "valid_pixels": 0.0,
    "creation_time": "2000-01-01-00:00:00",
    "sha_256": "...",
    "source_image": "./sample_image.tif",
    "source_image_sha_256": "...",
    "source_label": "./sample_label.tif",
    "source_label_sha_256": "...",
}
