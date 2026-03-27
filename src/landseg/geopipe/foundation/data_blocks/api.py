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
    - build_catalog: Run the end-to-end dataset workflow.
'''

# standard imports
import dataclasses
# local imports
import landseg.geopipe.foundation.common as common
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BlockBuildingConfig:
    '''Dataclass configuration for `build_catalog` pipeline.'''
    dev_image_fpath: str
    dev_label_fpath: str
    eval_image_fpath: str | None
    eval_label_fpath: str | None
    data_config_fpath: str
    dem_pad: int
    ignore_index: int

# -------------------------------Public Function-------------------------------
def build_blocks(
    world_grid: common.GridLayoutLike,
    config: BlockBuildingConfig,
    output_root: str,
    logger: utils.Logger,
    *,
    single_block_mode: bool = False,
    **kwargs
) -> None:
    '''
    Canonical data blocks building pipeline.

    This is current public interface for module `landseg.geopipe.blocks`
    that read input world grid (see protocol `GridLayoutLike` at common/)
    and data rasters to produce canonical data block files save as `.npz`
    files. A catalog JSON is also generated.

    Args:
        world_grid: Grid definition.
        config: Catalog building configuration.
        output_root: Root folder to store module artifacts.
        logger: A custom Logger class for logging purposes.
        single_block_mode: if True this pipeline only build and save a
            single block file, e.g., for an overfit test.
    '''

    # map model dev rasters to grid
    logger.log('INFO', 'Mapping rasters for model developement to grid')
    dev_ras_cfg = data_blocks.MappingConfig(
        input_img_fpath=config.dev_image_fpath,
        input_lbl_fpath=config.dev_label_fpath,
        output_root=f'{output_root}/model_dev/windows',
    )
    ras_windows = data_blocks.map_rasters(world_grid, dev_ras_cfg, logger)
    logger.log('INFO', 'Rasters for model developement mapped to grid')

    # block builder for model dev rasters
    builder_cfg = data_blocks.BuilderConfig(
        image_fpath=config.dev_image_fpath,
        label_fpath=config.dev_label_fpath,
        config_fpath=config.data_config_fpath,
        output_root=f'{output_root}/model_dev/',
        dem_pad_px=config.dem_pad,
        ignore_index=config.ignore_index
    )
    block_builder = data_blocks.BlockBuilder(ras_windows, builder_cfg, logger)

    # build just one block, e.g., for overfit test
    if single_block_mode:
        logger.log('INFO', 'Build a single block')
        output_dir = f'{output_root}/single/'
        block_builder.build_single_block(
            output_dir,
            valid_px_per= kwargs.get('valid_px_per', 0.8),
            monitor_head= kwargs.get('monitor_head', 'layer1'),
            need_all_classes= kwargs.get('need_all_classes', True)
        )
        return
    # build all model dev blocks
    logger.log('INFO', 'Build all model developement data blocks')
    new_blocks = block_builder.build_blocks()

    # create/update catalog and metadata JSON
    updated = data_blocks.CatalogUpdateContext(
        coords=new_blocks,
        source_image=config.dev_image_fpath,
        source_label=config.dev_label_fpath,
        mapped_grid_id=world_grid.gid
    )
    data_blocks.update_catalog(updated, f'{output_root}/model_dev/', logger)
    data_blocks.update_meta(updated, f'{output_root}/model_dev/', logger)

    # exit if evaluation rasters are not provided
    if not (config.eval_image_fpath and config.eval_label_fpath):
        logger.log('INFO', 'Evaluation holdout rasters not provided, exit')
        return

    # map evaluation rasters to grid
    logger.log('INFO', 'Mapping holdout raters for evaluation to grid')
    dev_ras_cfg = data_blocks.MappingConfig(
        input_img_fpath=config.eval_image_fpath,
        input_lbl_fpath=config.eval_label_fpath,
        output_root=f'{output_root}/eval_holdout/windows',
    )
    ras_windows = data_blocks.map_rasters(world_grid, dev_ras_cfg, logger)
    logger.log('INFO', 'Holdout raters for evaluation mapped to grid')

    # block builder for evaluation rasters
    builder_cfg = data_blocks.BuilderConfig(
        image_fpath=config.eval_image_fpath,
        label_fpath=config.eval_label_fpath,
        config_fpath=config.data_config_fpath,
        output_root=f'{output_root}/eval_holdout/',
        dem_pad_px=config.dem_pad,
        ignore_index=config.ignore_index
    )
    block_builder = data_blocks.BlockBuilder(ras_windows, builder_cfg, logger)
    logger.log('INFO', 'Build all holdout evaluation data blocks')
    new_blocks = block_builder.build_blocks()

    # create/update catalog and metadata JSON
    updated = data_blocks.CatalogUpdateContext(
        coords=new_blocks,
        source_image=config.eval_image_fpath,
        source_label=config.eval_label_fpath,
        mapped_grid_id=world_grid.gid
    )
    data_blocks.update_catalog(updated, f'{output_root}/eval_holdout/', logger)
    data_blocks.update_meta(updated, f'{output_root}/eval_holdout/', logger)
