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
    - prepare_data: Run the end-to-end dataprep workflow.
'''

# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.dataprep as dataprep
import landseg.dataprep.blockbuilder as blockbuilder
import landseg.dataprep.mapper as mapper
import landseg.dataprep.normalizer as normalizer
import landseg.dataprep.splitter as splitter
import landseg.grid as grid
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def prepare_data(
    world_grid: tuple[str, grid.GridLayout],
    input_config: configs.InputDataCfg,
    prep_config: configs.PrepDataCfg,
    logger: utils.Logger,
    **kwargs
) -> core.SchemaOneBlock | None:
    '''
    Run the dataprep workflow: map rasters, build/normalize blocks, split
    sets, and generate schema. Supports a single-block mode for testing.

    Args:
        world_grid: Target grid description.
        inputs_config: Config mapping for input data sources.
        artifact_config: Config mapping for artifact/cache/output paths.
        process_config: Config mapping for thresholds and scoring
            parameters.
        logger: Logger for progress and diagnostics.
        kwargs: Optional flags
            - rebuild_all (bool): Rebuild all stages. Defaults to False.
            - remap (bool): Remap raster windows. Defaults to
                `rebuild_all`.
            - rebuild_blocks (bool): Rebuild block caches. Defaults to
                `rebuild_all`.
            - renormalize (bool): Recompute image normalization. Defaults
                to `rebuild_all`.
            - rebuild_split (bool): Regenerate train/val splits. Defaults
                to `rebuild_all`.
            - build_a_block (bool): If True, produce a single block
                artifact and return its schema. Defaults to False.
            - block_fpath (str): Output path for the single block when
                build_a_block=True.

    Returns:
        (dict | None): Schema when build_a_block=True; otherwise None
            (schema persists as an artifact).
    '''

    # get flags from keyword arguments
    rebuild_all = kwargs.get('rebuild_all', False)
    remap = kwargs.get('remap', rebuild_all)
    rebuild_blks = kwargs.get('rebuild_blocks', rebuild_all)
    renorm = kwargs.get('renormalize', rebuild_all)
    rebuild_split = kwargs.get('rebuild_split', rebuild_all)

    # get a child logger
    logger = logger.get_child('dprep')

    # get pipeline configs from input
    cfg = _parse_configs(input_config, prep_config)

    # map rasters to world grid (alway map fit, map test if provided)
    mapper.map_rasters(world_grid[1], cfg, logger, remap=remap)

    # if single block mode - build and return the instance
    if kwargs.get('build_a_block', False):
        logger.log('INFO', 'Single data block preparation mode')
        block_fpath = kwargs.get('block_fpath')
        assert block_fpath, 'No block file path provided'
        block = blockbuilder.build_one_block(cfg, logger)
        block.save(block_fpath)
        return dataprep.build_schema_one_block(block_fpath, block)

    # build/normalize/split fit blocks
    blockbuilder.build_blocks('fit', cfg, logger, rebuild=rebuild_blks)
    normalizer.normalize_blocks('fit', cfg, logger, renormalize=renorm)
    splitter.split_blocks(cfg, logger, rebuild=rebuild_split)

    # # build/normalize test blocks if provided
    if cfg['test_input_img']:
        blockbuilder.build_blocks('test', cfg, logger, rebuild=rebuild_blks)
        normalizer.normalize_blocks('test', cfg, logger, renormalize=renorm)

    # generate schema
    dataprep.build_schema_full(world_grid, prep_config.output_dirpath, cfg)
    return None

# ------------------------------private  function------------------------------
def _parse_configs(
    input_config: configs.InputDataCfg,
    prep_config: configs.PrepDataCfg,
) -> dataprep.DataprepConfigs:
    '''Parse and consolidate input configs into a typed config.'''

    return_cfg: dataprep.DataprepConfigs = {
        # input - raw data paths
        'fit_input_img': input_config.filepaths.fit_image,
        'fit_input_lbl': input_config.filepaths.fit_label,
        'test_input_img': input_config.filepaths.test_image,
        'test_input_lbl': input_config.filepaths.test_label,
        'input_config': input_config.filepaths.config,
        # fit data artifacts
        'fit_windows': prep_config.fit_blocks.raster_windows,
        'fit_blks_dir': prep_config.fit_blocks.blocks_dir,
        'fit_all_blks': prep_config.fit_blocks.all_blocks,
        'fit_valid_blks': prep_config.fit_blocks.valid_blocks,
        'fit_img_stats': prep_config.fit_post_blocks.image_stats,
        'lbl_count_global': prep_config.fit_post_blocks.label_count_global,
        'blk_scores': prep_config.fit_post_blocks.block_scores,
        'train_blks': prep_config.fit_post_blocks.train_blocks_split,
        'val_blks': prep_config.fit_post_blocks.val_blocks_split,
        'lbl_count_train': prep_config.fit_post_blocks.label_count_train,
        # test data artifacts
        'test_windows': prep_config.test_blocks.raster_windows,
        'test_blks_dir': prep_config.test_blocks.blocks_dir,
        'test_all_blks': prep_config.test_blocks.all_blocks,
        'test_valid_blks': prep_config.test_blocks.valid_blocks,
        'test_img_stats': prep_config.test_post_blocks.image_stats,
        # thresholds
        'blk_thres_fit': prep_config.threshold.blk_thres_fit,
        'blk_thres_test': prep_config.threshold.blk_thres_test,
        # scoring
        'score_head': prep_config.scoring.head,
        'score_alpha': prep_config.scoring.alpha,
        'score_beta': prep_config.scoring.beta,
        'score_epsilon': prep_config.scoring.epsilon,
        'score_reward': tuple(prep_config.scoring.reward),
    }
    return return_cfg
