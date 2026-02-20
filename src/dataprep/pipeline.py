'''doc'''

# standard imports
import os
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
    dataprep.map_rasters(world_grid, cfg, logger)
    # build fit blocks
    dataprep.build_data_blocks('fit', cfg, logger)
    # normalize fit blocks
    dataprep.normalize_data_blocks('fit', cfg, logger)
    # split fit blocks
    dataprep.split_blocks(cfg, logger)

    # map test raster to world grid if provided
    if cfg['test_input_img']:
        dataprep.map_rasters(world_grid, cfg, logger)
        # build test blocks
        dataprep.build_data_blocks('test', cfg, logger)
        # normalize test blocks
        dataprep.normalize_data_blocks('test', cfg, logger)

    # generate schema
    data_cache_root = f'{artifact_config['cache']/inputs_config['name']}'
    dataprep.build_schema(data_cache_root, cfg)

def _parse_configs(
    input_data_config: alias.ConfigType,
    output_artifact_config: alias.ConfigType,
    process_config: alias.ConfigType,
) -> dataprep.DataprepConfigs:
    '''doc'''

    # config accessors
    input_cfg = utils.ConfigAccess(input_data_config)
    output_cfg = utils.ConfigAccess(output_artifact_config)
    proc_cfg = utils.ConfigAccess(process_config)

    return_cfg: dataprep.DataprepConfigs = {
        # input - raw data paths
        'fit_input_img': input_cfg.get_option('fit', 'image'),
        'fit_input_lbl': input_cfg.get_option('fit', 'label'),
        'test_input_img': input_cfg.get_option('test', 'image', default=None),
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
        'test_img_stats': output_cfg.get_option('test_image_stats'),
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
