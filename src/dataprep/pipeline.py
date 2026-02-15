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
    windows = dataprep.map_rasters(
        world_grid=world_grid,
        image_fpath=cfg['train_img'],
        label_fpath=cfg['train_lbl'],
        logger=logger
    )
    builder = dataprep.get_block_builder(windows, 'train', cfg, logger)
    builder.build_block_cache()
    builder.build_valid_block_index(cfg['train_px_thres'])

    # map inference rasters to world grid if provided
    if cfg['infer_img']:
        windows = dataprep.map_rasters(
            world_grid=world_grid,
            image_fpath=cfg['infer_img'],
            label_fpath=None,
            logger=logger
        )
        builder = dataprep.get_block_builder(windows, 'infer', cfg, logger)
        builder.build_block_cache()
        builder.build_valid_block_index(cfg['infer_px_thres'])

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
        'train_img': data_cfg.get_option('training', 'image', default=None),
        'train_lbl': data_cfg.get_option('training', 'label', default=None),
        'infer_img': data_cfg.get_option('inference', 'image', default=None),
        'input_config': data_cfg.get_option('config', default=None),
        'train_blks_dir': artifact_cfg.get_option('training_blocks_dir'),
        'train_all_blks': artifact_cfg.get_option('training_all_blocks'),
        'train_valid_blks': artifact_cfg.get_option('training_valid_blocks'),
        'infer_blks_dir': artifact_cfg.get_option('inference_blocks_dir'),
        'infer_all_blks': artifact_cfg.get_option('inference_all_blocks'),
        'train_px_thres': cache_cfg.get_option('threshold', 'train_blocks_px'),
        'infer_px_thres': cache_cfg.get_option('threshold', 'infer_blocks_px')
    }

    # sanity checks on required items
    assert return_cfg['train_img'] is not None
    assert return_cfg['input_config'] is not None

    return return_cfg
