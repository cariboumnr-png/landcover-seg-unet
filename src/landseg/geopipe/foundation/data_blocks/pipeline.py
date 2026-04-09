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

# pylint: disable=missing-function-docstring

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
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.utils as utils

# --------------------------------private types--------------------------------
class _PipelinePaths(typing.Protocol):
    '''Typed pipeline-specific paths container.'''
    @property
    def blocks(self) -> str:...
    @property
    def catalog(self) -> str:...
    @property
    def schema(self) -> str:...
    def mapped_window(self, gid: str) -> str:...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BlockBuildingParameters:
    '''Config container for the canonical block-building pipeline.'''
    image_fpath: str
    label_fpath: str | None
    data_config_fpath: str
    dem_pad: int
    ignore_index: int

# -------------------------------Public Function-------------------------------
def run_blocks_building(
    logger: utils.Logger,
    world_grid: geo_core.GridLayout,
    artfact_paths: _PipelinePaths,
    config: BlockBuildingParameters,
    *,
    policy: artifacts.LifecyclePolicy
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
    ras_windows = data_blocks.map_rasters_to_grid(
        world_grid,
        config.image_fpath,
        config.label_fpath,
        artfact_paths.mapped_window(world_grid.gid),
        policy=policy,
        logger=logger,
    )
    logger.log('INFO', 'Rasters mapped to input world grid')

    # block builder for model dev rasters
    builder_config = data_blocks.BlockBuilderConfig(
        output_root=artfact_paths.blocks,
        image_fpath=config.image_fpath,
        label_fpath=config.label_fpath,
        config_fpath=config.data_config_fpath,
        dem_pad_px=config.dem_pad,
        ignore_index=config.ignore_index,
        block_size=ras_windows.tile_shape
    )
    block_builder = data_blocks.BlockBuilder(
        ras_windows.image,
        ras_windows.label,
        builder_config,
        logger=logger,
    )

    # build all model dev blocks
    logger.log('INFO', 'Data blocks building finished')
    new_blocks = block_builder.build_blocks()

    # create/update catalog and metadata JSON
    updated = data_blocks.ManifestUpdateContext(
        updated_coords=new_blocks,
        source_image=config.image_fpath,
        source_label=config.label_fpath,
        mapped_grid_id=world_grid.gid,
        blocks_dir=artfact_paths.blocks,
        catalog_fpath=artfact_paths.catalog,
        schema_fpath=artfact_paths.schema
    )
    data_blocks.update_manifest(updated, policy=policy, logger=logger)
    logger.log('INFO', 'Data blocks catalog and schema updated')
