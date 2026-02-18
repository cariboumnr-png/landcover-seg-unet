'''doc'''

# local imports
import alias
import dataprep
import grid
import utils

def prepare_data(
    world_grid: grid.GridLayout,
    inputs_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    cache_config: alias.ConfigType,
    logger: utils.Logger
):
    '''doc'''

    # get a child logger
    logger = logger.get_child('dprep')

    # get pipeline configs from input
    cfg = _parse_configs(inputs_config, artifact_config, cache_config)

    # map training rasters to world grid
    windows = dataprep.map_rasters(world_grid, 'fit', cfg, logger)
    builder = dataprep.get_block_builder(windows, 'fit', cfg, logger)
    builder.build_block_cache()
    builder.build_valid_block_index(cfg['fit_blk_thres'])

    # map test raster to world grid if provided
    if cfg['test_input_img']:
        windows = dataprep.map_rasters(world_grid, 'test', cfg, logger)
        builder = dataprep.get_block_builder(windows, 'test', cfg, logger)
        builder.build_block_cache()
        builder.build_valid_block_index(cfg['test_blk_thres'])

def _parse_configs(
    data_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    cache_config: alias.ConfigType,
) -> dataprep.DataprepConfigs:
    '''doc'''

    # config accessors
    data_cfg = utils.ConfigAccess(data_config)
    artifact_cfg = utils.ConfigAccess(artifact_config)
    cache_cfg = utils.ConfigAccess(cache_config)

    return_cfg: dataprep.DataprepConfigs = {
        # inputs
        'fit_input_img': data_cfg.get_option('fit', 'image'),
        'fit_input_lbl': data_cfg.get_option('fit', 'label'),
        'test_input_img': data_cfg.get_option('test', 'image', default=None),
        'input_config': data_cfg.get_option('config'),
        # artifacts
        'fit_blks_dir': artifact_cfg.get_option('fit_blocks_dir'),
        'fit_all_blks': artifact_cfg.get_option('fit_all_blocks'),
        'fit_valid_blks': artifact_cfg.get_option('fit_valid_blocks'),
        'test_blks_dir': artifact_cfg.get_option('test_blocks_dir'),
        'test_all_blks': artifact_cfg.get_option('test_all_blocks'),
        # thresholds
        'fit_blk_thres': cache_cfg.get_option('threshold', 'train_blocks_px'),
        'test_blk_thres': cache_cfg.get_option('threshold', 'infer_blocks_px'),
        # scoring
        'score_head': cache_cfg.get_option('scoring', 'head'),
        'score_alpha': cache_cfg.get_option('scoring', 'alpha'),
        'score_beta': cache_cfg.get_option('scoring', 'beta'),
        'score_epsilon': cache_cfg.get_option('scoring', 'epsilon'),
        'score_reward': tuple(cache_cfg.get_option('scoring', 'reward')),
    }

    # sanity checks on required items
    assert return_cfg['fit_input_img'] is not None
    assert return_cfg['input_config'] is not None

    return return_cfg
