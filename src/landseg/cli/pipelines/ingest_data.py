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

Prepares the world grid, materializes domain knowledge, and builds
the immutable raw block catalogue for later experiments.
'''

# local imports
import landseg.configs as configs
import landseg.geopipe.foundation as foundation
import landseg.utils as utils

def ingest(config: configs.RootConfig):
    '''
    Run the ingestion pipeline.

    Steps:
    1) Build or load the world grid (`geopipe.foundation`).
    2) Prepare domain knowledge aligned to the grid.
    3) Build raw `.npz` data blocks and update `catalog.json` /
    `metadata.json`.

    Args:
        config: RootConfig with foundation settings.
    '''

    # init a logger
    logger = utils.Logger('ingest', f'{config.exp_root}/ingest.log')

    # config aliases
    grid_cfg = config.foundation.grid
    domain_cfg = config.foundation.domains
    datablocks_cfg = config.foundation.datablocks
    out_root = config.foundation.output_dpath

    # world grid
    _config = foundation.GridExtentConfig(
        mode=grid_cfg.mode, # type: ignore
        crs=grid_cfg.crs,
        ref_fpath=grid_cfg.extent.filepath,
        origin=grid_cfg.extent.origin,
        pixel_size=grid_cfg.extent.pixel_size,
        grid_extent=grid_cfg.extent.grid_extent,
        grid_shape=grid_cfg.extent.grid_shape
    )
    grid_gen_config = foundation.GridGenerationConfig(
        output_dir=f'{out_root}/world_grids',
        tile_size=(grid_cfg.tile_size.row, grid_cfg.tile_size.col),
        tile_overlap=(grid_cfg.tile_overlap.row, grid_cfg.tile_overlap.col)
    )
    grid = foundation.prep_world_grid(_config, grid_gen_config, logger)

    # domains
    _config = foundation.DomainMappingConfig(
        file_list=[(d.path, d.index_base) for d in domain_cfg.files],
        valid_threshold=domain_cfg.valid_threshold,
        target_variance=domain_cfg.target_variance,
        output_dir=f'{out_root}/domain_knowledge',
    )
    foundation.prepare_domain(grid, _config, logger)

    # datablocks building
    _config = foundation.BlockBuildingConfig(
        dev_image_fpath=datablocks_cfg.filepaths.dev_image,
        dev_label_fpath=datablocks_cfg.filepaths.dev_label,
        eval_image_fpath=datablocks_cfg.filepaths.test_image,
        eval_label_fpath=datablocks_cfg.filepaths.test_label,
        data_config_fpath=datablocks_cfg.filepaths.config,
        dem_pad=datablocks_cfg.general.image_dem_pad,
        ignore_index=datablocks_cfg.general.ignore_index,
    )
    blocks_dir = f'{out_root}/data_blocks'
    foundation.build_blocks(grid, _config, blocks_dir, logger)
