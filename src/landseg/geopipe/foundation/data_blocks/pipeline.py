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
Canonical data-block construction pipeline.

Maps input rasters onto a pre-built world grid, materializes immutable
raw data blocks, and maintains the associated catalog and dataset
metadata. This pipeline does **not** perform dataset splitting or
normalization; it produces experiment-agnostic artifacts intended for
reuse across downstream workflows.

Public API:
    - build_blocks: Build raw data blocks and update catalog/metadata.
'''

# standard imports
import dataclasses
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BlockBuildingParameters:
    '''Config container for the canonical block-building pipeline.'''
    dev_image_fpath: str
    dev_label_fpath: str
    test_image_fpath: str | None
    test_label_fpath: str | None
    data_config_fpath: str
    dem_pad: int
    ignore_index: int

# -------------------------------Public Function-------------------------------
def run_blocks_building(
    world_grid: geo_core.GridLayout,
    config: BlockBuildingParameters,
    output_root: str,
    logger: utils.Logger,
    *,
    single_block_mode: bool = False,
    **kwargs
) -> str | None:
    '''
    Build canonical data blocks from rasters aligned to a world grid.

    This function is the public entrypoint for constructing immutable
    `.npz` block artifacts and their associated `catalog.json` and
    `metadata.json`. Blocks are built directly from raster windows
    without normalization or dataset splitting.

    Workflow:
    1) Map development rasters to the world grid.
    2) Build raw data blocks and update the catalog.
    3) Optionally repeat steps (1-2) for test holdout rasters.
    4) Optionally build a single block for debugging or overfit runs.

    Args:
        world_grid: World grid definition used to locate raster windows.
        config: Configuration for block building inputs and parameters.
        output_root: Root directory where block artifacts are written.
        logger: Logger instance used for progress and status reporting.
        single_block_mode: If True, build and persist only one valid
            block (e.g., for overfit or debugging workflows).
        **kwargs: Optional overrides for single-block mode behavior
            (e.g., validity threshold, monitor head, output path).

    Returns:
        Path to the saved block when `single_block_mode` is enabled;
        otherwise `None`.
    '''

    # get a child logger
    logger = logger.get_child('dblks')

    # map model dev rasters to grid
    logger.log('INFO', 'Mapping rasters for model developement to grid')
    ras_windows = data_blocks.map_rasters_to_grid(
        world_grid,
        (config.dev_image_fpath, config.dev_label_fpath),
        logger,
        artifacts_dir=f'{output_root}/model_dev/windows',
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )
    logger.log('INFO', 'Rasters for model developement mapped to grid')

    # block builder for model dev rasters
    builder_cfg = data_blocks.BlockBuilderConfig(
        image_fpath=config.dev_image_fpath,
        label_fpath=config.dev_label_fpath,
        config_fpath=config.data_config_fpath,
        output_root=f'{output_root}/model_dev/',
        dem_pad_px=config.dem_pad,
        ignore_index=config.ignore_index,
        block_size=ras_windows.tile_shape
    )
    block_builder = data_blocks.BlockBuilder(
        ras_windows.image,
        ras_windows.label,
        builder_cfg,
        logger
    )

    # build just one block, e.g., for overfit test
    if single_block_mode:
        logger.log('INFO', 'Build a single block')
        return block_builder.build_single_block(
            save_dpath=kwargs.get('save_dpath', f'{output_root}/single_block'),
            valid_px_per=kwargs.get('valid_px_per', 0.8),
            monitor_head=kwargs.get('monitor_head', 'base'),
            need_all_classes=kwargs.get('need_all_classes', True)
        )
    # build all model dev blocks
    logger.log('INFO', 'Build all model developement data blocks')
    new_blocks = block_builder.build_blocks()

    # create/update catalog and metadata JSON
    updated = data_blocks.ManifestUpdateContext(
        updated_coords=new_blocks,
        source_image=config.dev_image_fpath,
        source_label=config.dev_label_fpath,
        mapped_grid_id=world_grid.gid
    )
    data_blocks.update_manifest(
        updated,
        logger,
        artifacts_dir=f'{output_root}/model_dev/',
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )

    # exit if test rasters are not provided
    if not (config.test_image_fpath and config.test_label_fpath):
        logger.log('INFO', 'Evaluation holdout rasters not provided, exit')
        return None

    # map test rasters to grid
    logger.log('INFO', 'Mapping holdout raters for test to grid')
    ras_windows = data_blocks.map_rasters_to_grid(
        world_grid,
        (config.test_image_fpath, config.test_label_fpath),
        logger,
        artifacts_dir=f'{output_root}/test_holdout/windows',
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )
    logger.log('INFO', 'Holdout raters for test mapped to grid')

    # block builder for test rasters
    builder_cfg = data_blocks.BlockBuilderConfig(
        image_fpath=config.test_image_fpath,
        label_fpath=config.test_label_fpath,
        config_fpath=config.data_config_fpath,
        output_root=f'{output_root}/test_holdout/',
        dem_pad_px=config.dem_pad,
        ignore_index=config.ignore_index,
        block_size=ras_windows.tile_shape
    )
    block_builder = data_blocks.BlockBuilder(
        ras_windows.image,
        ras_windows.label,
        builder_cfg,
        logger
    )
    logger.log('INFO', 'Build all holdout test data blocks')
    new_blocks = block_builder.build_blocks()

    # create/update catalog and metadata JSON
    updated = data_blocks.ManifestUpdateContext(
        updated_coords=new_blocks,
        source_image=config.test_image_fpath,
        source_label=config.test_label_fpath,
        mapped_grid_id=world_grid.gid
    )
    data_blocks.update_manifest(
        updated,
        logger,
        artifacts_dir=f'{output_root}/test_holdout/',
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )
    return None
