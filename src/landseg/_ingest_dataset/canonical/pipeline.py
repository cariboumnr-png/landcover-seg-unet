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

# standard imports
import dataclasses
# local imports
import landseg.core as core
import landseg._ingest_dataset.canonical as canonical
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class CatalogueInputs:
    '''doc'''
    fit_image_fpath: str
    fit_label_fpath: str
    test_image_fpath: str | None
    test_label_fpath: str | None
    data_config_fpath: str

# -------------------------------Public Function-------------------------------
def build_catalogue(
    world_grid: core.GridLayoutLike,
    input_fpaths: CatalogueInputs,
    output_root: str,
    logger: utils.Logger,
    *,
    single_block_mode: bool = False
):
    '''Test new pipeline.'''

    # align fit rasters to grid
    fit_rasters_config = canonical.AlignmentConfig(
        input_img_fpath=input_fpaths.fit_image_fpath,
        input_lbl_fpath=input_fpaths.fit_label_fpath,
        output_windows_dpath=f'{output_root}/fit/windows',
    )
    canonical.align_rasters(world_grid, fit_rasters_config, logger)

    # get a block builder instance from configuration
    fit_block_build_config = canonical.BuilderConfig(
        image_fpath=input_fpaths.fit_image_fpath,
        label_fpath=input_fpaths.fit_label_fpath,
        config_fpath=input_fpaths.data_config_fpath,
        catalog_root=f'{output_root}/fit/',
        grid_id=world_grid.gid,
        dem_pad_px=8,
        ignore_index=255
    )
    block_builder = canonical.BlockBuilder(fit_block_build_config, logger)

    # build just one block, e.g., for overfit test
    if single_block_mode:
        block_builder.build_single_block('./experiment/results/overfit_test')
        return
    # build all fit blocks
    block_builder.build_blocks()

    # exit if test rasters are not provided
    if not (input_fpaths.test_image_fpath and input_fpaths.test_label_fpath):
        return

    # align test rasters to grid
    test_rasters_config = canonical.AlignmentConfig(
        input_img_fpath=input_fpaths.test_image_fpath,
        input_lbl_fpath=input_fpaths.test_label_fpath,
        output_windows_dpath=f'{output_root}/test/windows',
    )
    canonical.align_rasters(world_grid, test_rasters_config, logger)

    # build test blocks
    test_block_build_config = canonical.BuilderConfig(
        image_fpath=input_fpaths.test_image_fpath,
        label_fpath=input_fpaths.test_label_fpath,
        config_fpath=input_fpaths.data_config_fpath,
        catalog_root=f'{output_root}/test/',
        grid_id=world_grid.gid,
        dem_pad_px=8,
        ignore_index=255
    )
    block_builder = canonical.BlockBuilder(test_block_build_config, logger)
    block_builder.build_blocks()
