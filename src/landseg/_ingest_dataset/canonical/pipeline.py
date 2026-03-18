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
Data preparation pipeline that maps rasters to a grid, builds and
normalizes blocks, splits datasets, and persists a versioned schema.

Public APIs:
    - build_catalogue_test: Run the end-to-end dataset workflow.
'''

# local imports
import landseg.configs as configs
import landseg.core as core
import landseg._ingest_dataset.canonical as canonical
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def build_catalogue_test(
    world_grids: list[core.GridLayoutLike],
    config: configs.RootConfig,
    logger: utils.Logger,
    **kwargs
):
    '''Test new pipeline.'''

    root = './experiment/artifacts/data_cache/branch_test'

    # align rasters to world grids
    for grid in world_grids:
        # fit rasters
        fit_rasters_config: canonical.AlignmentConfig = {
            'input_img_fpath': config.inputs.data.filepaths.fit_image,
            'input_lbl_fpath': config.inputs.data.filepaths.fit_label,
            'output_windows_dpath': f'{root}/fit/windows',
        }
        canonical.align_rasters(grid, fit_rasters_config, logger)
        # test rasters
        test_rasters_config: canonical.AlignmentConfig = {
            'input_img_fpath': config.inputs.data.filepaths.test_image,
            'input_lbl_fpath': config.inputs.data.filepaths.test_label,
            'output_windows_dpath': f'{root}/test/windows',
        }
        canonical.align_rasters(grid, test_rasters_config, logger)

    # build fit blocks to selected grid
    fit_block_build_config: canonical.BuilderConfig = {
        'image_fpath': config.inputs.data.filepaths.fit_image,
        'label_fpath': config.inputs.data.filepaths.fit_label,
        'config_fpath': config.inputs.data.filepaths.config,
        'catalog_root': f'{root}/fit/',
        'grid_id': 'grid_row_256_128_col_256_128',
        'dem_pad_px': 8,
        'ignore_index': 255
    }
    block_builder = canonical.BlockBuilder(fit_block_build_config, logger)
    # one block mode (from fit blocks)
    if kwargs.get('build_a_block', False):
        block_builder.build_single_block('./experiment/results/overfit_test')
        return
    # all fit blocks
    block_builder.build_blocks()

    # build test blocks (currently requires a zero-overlap tilling)
    test_block_build_config: canonical.BuilderConfig = {
        'image_fpath': config.inputs.data.filepaths.test_image,
        'label_fpath': config.inputs.data.filepaths.test_label,
        'config_fpath': config.inputs.data.filepaths.config,
        'catalog_root': f'{root}/test/',
        'grid_id': 'grid_row_256_0_col_256_0',
        'dem_pad_px': 8,
        'ignore_index': 255
    }
    block_builder = canonical.BlockBuilder(test_block_build_config, logger)
    block_builder.build_blocks()
