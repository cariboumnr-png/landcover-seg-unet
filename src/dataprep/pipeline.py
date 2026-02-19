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
    proc_config: alias.ConfigType,
    logger: utils.Logger
):
    '''doc'''

    # get a child logger
    logger = logger.get_child('dprep')

    # get pipeline configs from input
    cfg = _parse_configs(inputs_config, artifact_config, proc_config)

    # map fit rasters to world grid
    windows = dataprep.map_rasters(world_grid, 'fit', cfg, logger)
    # build fit blocks
    builder = dataprep.get_block_builder(windows, 'fit', cfg, logger)
    builder.build_block_cache()
    builder.build_valid_block_index(cfg['blk_thres_fit'])
    # normalize fit blocks
    dataprep.normalize_data_blocks('fit', cfg, logger)
    # split fit blocks
    dataprep.split_blocks(cfg, logger)

    # map test raster to world grid if provided
    if cfg['test_input_img']:
        windows = dataprep.map_rasters(world_grid, 'test', cfg, logger)
        # build test blocks
        builder = dataprep.get_block_builder(windows, 'test', cfg, logger)
        builder.build_block_cache()
        builder.build_valid_block_index(cfg['blk_thres_test'])
        # normalize test blocks
        dataprep.normalize_data_blocks('test', cfg, logger)

def _parse_configs(
    data_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    proc_config: alias.ConfigType,
) -> dataprep.DataprepConfigs:
    '''doc'''

    # config accessors
    data_cfg = utils.ConfigAccess(data_config)
    artifact_cfg = utils.ConfigAccess(artifact_config)
    proc_cfg = utils.ConfigAccess(proc_config)

    return_cfg: dataprep.DataprepConfigs = {
        # input - raw data paths
        'fit_input_img': data_cfg.get_option('fit', 'image'),
        'fit_input_lbl': data_cfg.get_option('fit', 'label'),
        'test_input_img': data_cfg.get_option('test', 'image', default=None),
        'input_config': data_cfg.get_option('config'),
        # output - artifact paths
        'fit_blks_dir': artifact_cfg.get_option('fit_blocks_dir'),
        'fit_all_blks': artifact_cfg.get_option('fit_all_blocks'),
        'fit_valid_blks': artifact_cfg.get_option('fit_valid_blocks'),
        'fit_img_stats': artifact_cfg.get_option('fit_image_stats'),
        'lbl_count_global': artifact_cfg.get_option('label_count_global'),
        'blk_scores': artifact_cfg.get_option('block_scores'),
        'train_blks': artifact_cfg.get_option('train_blocks_split'),
        'val_blks': artifact_cfg.get_option('val_blocks_split'),
        'lbl_count_train': artifact_cfg.get_option('label_count_train'),
        'test_blks_dir': artifact_cfg.get_option('test_blocks_dir'),
        'test_all_blks': artifact_cfg.get_option('test_all_blocks'),
        'test_img_stats': artifact_cfg.get_option('test_image_stats'),
        # thresholds
        'blk_thres_fit': proc_cfg.get_option('threshold', 'train_blocks_px'),
        'blk_thres_test': proc_cfg.get_option('threshold', 'infer_blocks_px'),
        # scoring
        'score_head': proc_cfg.get_option('scoring', 'head'),
        'score_alpha': proc_cfg.get_option('scoring', 'alpha'),
        'score_beta': proc_cfg.get_option('scoring', 'beta'),
        'score_epsilon': proc_cfg.get_option('scoring', 'epsilon'),
        'score_reward': tuple(proc_cfg.get_option('scoring', 'reward')),
    }

    # sanity checks on required items
    assert return_cfg['fit_input_img'] is not None
    assert return_cfg['input_config'] is not None

    return return_cfg
