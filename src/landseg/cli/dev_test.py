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
import landseg.geopipe.datamake as datamake
import landseg.utils as utils

def catalogue(config: configs.RootConfig):
    '''Catalogue pipeline.'''

    logger = utils.Logger('test', './test.log')

    # world grid
    grid_ext_config = datamake.GridExtentConfig(
        mode=config.inputs.extent.mode, # type: ignore
        ref_fpath='./experiment/input/extent_reference/on_3161_30m.tif',
        crs=config.inputs.extent.crs,
        origin=config.inputs.extent.inputs.origin,
        pixel_size=config.inputs.extent.inputs.pixel_size,
        grid_extent=config.inputs.extent.inputs.grid_extent,
        grid_shape=config.inputs.extent.inputs.grid_shape
    )
    grid_gen_config = datamake.GridGenerationConfig(
        output_dir=config.prep.grid.output_dirpath,
        tile_size=(256, 256),
        tile_overlap=(0, 0),
    )
    grid = datamake.prep_world_grid(grid_ext_config, grid_gen_config, logger)

    # domains
    domain_config = datamake.DomainMappingConfig(
        source_dir=config.inputs.domain.input_dirpath,
        file_list=config.inputs.domain.files, # type: ignore
        output_dir=config.prep.domain.output_dirpath,
        valid_threshold=config.prep.domain.valid_threshold,
        target_variance=config.prep.domain.target_variance
    )
    datamake.prepare_domain(grid, domain_config, logger)

    # datablocks building
    blocks_config = datamake.BlockBuildingConfig(
        dev_image_fpath=config.inputs.data.filepaths.fit_image,
        dev_label_fpath=config.inputs.data.filepaths.fit_label,
        eval_image_fpath=config.inputs.data.filepaths.test_image,
        eval_label_fpath=config.inputs.data.filepaths.test_label,
        data_config_fpath=config.inputs.data.filepaths.config,
        dem_pad=8,
        ignore_index=255
    )
    blocks_dir = './experiment/artifacts/data_cache/branch_test'
    datamake.build_blocks(grid, blocks_config, blocks_dir, logger)

    # # datablocks partition
    # part_config = geopipe.PartitionConfig(
    #     val_test_ratios=(0.1, 0),
    #     buffer_step=1,
    #     reward_ratios={2: 5.0, 4: 5.0},
    #     scoring_alpha=1.0,
    #     scoring_beta=0.8,
    #     max_skew_rate=5.0,
    #     block_spec=(256, 128)
    # )
    # geopipe.partition_blocks()

    # # parse from catalog
    # work_catalog = {k: v for k, v in catalog.items() if v['valid_px']}
    # catalog_counts = {k: v['class_count'] for k, v in work_catalog.items()}

    # # from base grid (no overlap)
    # block_size = config.block_spec[0]
    # base_catalog = {
    #     k: v for k, v in work_catalog.items()
    #     if all(i % block_size == 0 for i in v['loc_col_row']) # both divisible
    # }
    # base_counts = {k: v['class_count'] for k, v in base_catalog.items()}

    # # partition data blocks
    # partitions = materialized.partition_blocks(
    #     base_counts,
    #     catalog_counts,
    #     config,
    #     logger,
    #     block_spec=config.block_spec
    # )

    # # normalize
    # train_blocks = [
    #     v['file_path'] for k, v in catalog.items() if k in partitions.train
    # ]
    # all_coords = partitions.train + partitions.val + partitions.test
    # all_blocks =  [
    #     v['file_path'] for k, v in catalog.items() if k in all_coords
    # ]
    # materialized.build_normalized_blocks(
    #     train_blocks,
    #     all_blocks,
    #     f'{output_root}/blocks'
    # )
