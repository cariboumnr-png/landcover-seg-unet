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

# standard imports
import typing
# local imports
import landseg.alias as alias
import landseg.dataprep as dataprep
import landseg.dataprep.blockbuilder as blockbuilder
import landseg.dataprep.mapper as mapper
import landseg.dataprep.normalizer as normalizer
import landseg.dataprep.splitter as splitter
import landseg.grid as grid
import landseg.utils as utils

def prepare_data(
    world_grid: tuple[str, grid.GridLayout],
    inputs_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    process_config: alias.ConfigType,
    logger: utils.Logger,
    **kwargs
) -> dict[str, typing.Any] | None:
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
    cfg = _parse_configs(inputs_config, artifact_config, process_config)

    # map rasters to world grid (alway map fit, map test if provided)
    mapper.map_rasters(world_grid[1], cfg, logger, remap=remap)

    # if single block mode - build and return the instance
    if kwargs.get('build_a_block', False):
        logger.log('INFO', 'Single data block preparation mode')
        block_fpath = kwargs.get('block_fpath')
        assert block_fpath, 'No block file path provided'
        block = blockbuilder.build_a_block(cfg, logger)
        block.save(block_fpath)
        return dataprep.schema_from_a_block(block_fpath, block)

    # build/normalize/split fit blocks
    blockbuilder.build_blocks('fit', cfg, logger, rebuild=rebuild_blks)
    normalizer.normalize_blocks('fit', cfg, logger, renormalize=renorm)
    splitter.split_blocks(cfg, logger, rebuild=rebuild_split)

    # # build/normalize test blocks if provided
    if cfg['test_input_img']:
        blockbuilder.build_blocks('test', cfg, logger, rebuild=rebuild_blks)
        normalizer.normalize_blocks('test', cfg, logger, renormalize=renorm)

    # generate schema
    data_cache_root = f'{artifact_config["cache"]}/{inputs_config["name"]}'
    dataprep.build_schema(world_grid, data_cache_root, cfg)
    return None

def _parse_configs(
    input_data_config: alias.ConfigType,
    output_artifact_config: alias.ConfigType,
    process_config: alias.ConfigType,
) -> dataprep.DataprepConfigs:
    '''Parse and consolidate input configs into a typed config.'''

    # config accessors
    input_cfg = utils.ConfigAccess(input_data_config)
    output_cfg = utils.ConfigAccess(output_artifact_config)
    proc_cfg = utils.ConfigAccess(process_config)

    return_cfg: dataprep.DataprepConfigs = {
        # input - raw data paths
        'fit_input_img': input_cfg.get_option('fit', 'image'),
        'fit_input_lbl': input_cfg.get_option('fit', 'label'),
        'test_input_img': input_cfg.get_option('test', 'image', default=None),
        'test_input_lbl': input_cfg.get_option('test', 'label', default=None),
        'input_config': input_cfg.get_option('config'),
        # output - artifact paths
        'fit_windows': output_cfg.get_option('fit_raster_windows'),
        'fit_blks_dir': output_cfg.get_option('fit_blocks_dir'),
        'fit_all_blks': output_cfg.get_option('fit_all_blocks'),
        'fit_valid_blks': output_cfg.get_option('fit_valid_blocks'),
        'fit_img_stats': output_cfg.get_option('fit_image_stats'),
        'lbl_count_global': output_cfg.get_option('label_count_global'),
        'blk_scores': output_cfg.get_option('block_scores'),
        'train_blks': output_cfg.get_option('train_blocks_split'),
        'val_blks': output_cfg.get_option('val_blocks_split'),
        'lbl_count_train': output_cfg.get_option('label_count_train'),
        'test_windows': output_cfg.get_option('test_raster_windows'),
        'test_blks_dir': output_cfg.get_option('test_blocks_dir'),
        'test_all_blks': output_cfg.get_option('test_all_blocks'),
        'test_valid_blks': output_cfg.get_option('test_valid_blocks'),
        'test_img_stats': output_cfg.get_option('test_image_stats'),
        # thresholds
        'blk_thres_fit': proc_cfg.get_option('threshold', 'blk_thres_fit'),
        'blk_thres_test': proc_cfg.get_option('threshold', 'blk_thres_test'),
        # scoring
        'score_head': proc_cfg.get_option('scoring', 'head'),
        'score_alpha': proc_cfg.get_option('scoring', 'alpha'),
        'score_beta': proc_cfg.get_option('scoring', 'beta'),
        'score_epsilon': proc_cfg.get_option('scoring', 'epsilon'),
        'score_reward': tuple(proc_cfg.get_option('scoring', 'reward')),
    }

    # sanity checks on required items and return
    assert return_cfg['fit_input_img'] is not None
    assert return_cfg['input_config'] is not None
    return return_cfg
