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
import landseg.geopipe.foundation as foundation
import landseg.geopipe.transform as transform
import landseg.utils as utils

def test_run(config: configs.RootConfig):
    '''test'''

    catalogue(config)
    prep_train_data_test(config)

def catalogue(config: configs.RootConfig):
    '''Catalogue pipeline.'''

    logger = utils.Logger('test', './test.log')

    # world grid
    grid_ext_config = foundation.GridExtentConfig(
        mode=config.inputs.extent.mode, # type: ignore
        ref_fpath='./experiment/input/extent_reference/on_3161_30m.tif',
        crs=config.inputs.extent.crs,
        origin=config.inputs.extent.inputs.origin,
        pixel_size=config.inputs.extent.inputs.pixel_size,
        grid_extent=config.inputs.extent.inputs.grid_extent,
        grid_shape=config.inputs.extent.inputs.grid_shape
    )
    grid_gen_config = foundation.GridGenerationConfig(
        output_dir=config.prep.grid.output_dirpath,
        tile_size=(256, 256),
        tile_overlap=(128, 128),
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
        dev_image_fpath=config.inputs.data.filepaths.fit_image,
        dev_label_fpath=config.inputs.data.filepaths.fit_label,
        eval_image_fpath=config.inputs.data.filepaths.test_image,
        eval_label_fpath=config.inputs.data.filepaths.test_label,
        data_config_fpath=config.inputs.data.filepaths.config,
        dem_pad=8,
        ignore_index=255
    )
    blocks_dir = './experiment/artifacts/foundation'
    foundation.build_blocks(grid, blocks_config, blocks_dir, logger)

def prep_train_data_test(config: configs.RootConfig):
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
