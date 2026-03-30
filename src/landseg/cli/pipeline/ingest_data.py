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
Data ingestion pipeline.
'''

# local imports
import landseg.configs as configs
import landseg.geopipe.foundation as foundation
import landseg.utils as utils

def ingest(config: configs.RootConfig):
    '''Data ingestion pipeline.'''

    # init a logger
    logger = utils.Logger('ingest', f'{config.exp_root}/ingest.log')

    # config aliases
    extent = config.inputs.extent
    grid = config.prep.grid

    # world grid
    grid_ext_config = foundation.GridExtentConfig(
        mode=extent.mode, # type: ignore
        crs=extent.crs,
        ref_fpath=extent.inputs.filepath,
        origin=extent.inputs.origin,
        pixel_size=extent.inputs.pixel_size,
        grid_extent=extent.inputs.grid_extent,
        grid_shape=extent.inputs.grid_shape
    )
    grid_gen_config = foundation.GridGenerationConfig(
        output_dir=grid.output_dirpath,
        tile_size=(grid.tile_size.row, grid.tile_size.col),
        tile_overlap=(grid.tile_overlap.row, grid.tile_overlap.col)
    )
    grid = foundation.prep_world_grid(grid_ext_config, grid_gen_config, logger)

    # domains
    domain_config = foundation.DomainMappingConfig(
        source_dir=config.inputs.domain.input_dirpath,
        file_list=config.inputs.domain.files, # type: ignore
        output_dir=config.prep.domain.output_dirpath,
        valid_threshold=config.prep.domain.valid_threshold,
        target_variance=config.prep.domain.target_variance
    )
    foundation.prepare_domain(grid, domain_config, logger)

    # datablocks building
    blocks_config = foundation.BlockBuildingConfig(
        dev_image_fpath=config.inputs.data.filepaths.dev_image,
        dev_label_fpath=config.inputs.data.filepaths.dev_label,
        eval_image_fpath=config.inputs.data.filepaths.test_image,
        eval_label_fpath=config.inputs.data.filepaths.test_label,
        data_config_fpath=config.inputs.data.filepaths.config,
        dem_pad=config.prep.data.general.image_dem_pad,
        ignore_index=config.prep.data.general.ignore_index,
    )
    blocks_dir = config.prep.data.artifacts.foundation
    foundation.build_blocks(grid, blocks_config, blocks_dir, logger)
