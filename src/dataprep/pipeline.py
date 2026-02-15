'''doc'''

# standard imports
import typing
# local imports
import alias
import dataprep
import grid
import utils

class DataprepConfigs(typing.TypedDict):
    '''doc'''
    # input
    train_img: str
    train_lbl: str | None
    infer_img: str | None
    input_config: str
    # artifacts
    train_blks_dir: str
    train_all_blks: str
    train_valid_blks: str
    infer_blks_dir: str
    infer_all_blks: str
    # thresholds
    train_px_thres: float
    infer_px_thres: float

def prepare_data(
    world_grid: grid.GridLayout,
    data_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    cache_config: alias.ConfigType,
    logger: utils.Logger
):
    '''doc'''

    # get a child logger
    logger = logger.get_child('dprep')

    # get pipeline configs from input
    cfg = _parse_configs(data_config, artifact_config, cache_config)

    # map training rasters to world grid
    windows = dataprep.map_rasters(
        world_grid=world_grid,
        image_fpath=cfg['train_img'],
        label_fpath=cfg['train_lbl'],
        logger=logger
    )
    _build_blocks_by_mode(windows, 'train', cfg, logger)

    # map inference rasters to world grid if provided
    if cfg['infer_img']:
        windows = dataprep.map_rasters(
            world_grid=world_grid,
            image_fpath=cfg['infer_img'],
            label_fpath=None,
            logger=logger
        )
        _build_blocks_by_mode(windows, 'infer', cfg, logger)

def _parse_configs(
    data_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    cache_config: alias.ConfigType,
) -> DataprepConfigs:
    '''doc'''

    # config accessors
    data_cfg = utils.ConfigAccess(data_config)
    artifact_cfg = utils.ConfigAccess(artifact_config)
    cache_cfg = utils.ConfigAccess(cache_config)

    return_cfg: DataprepConfigs = {
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

def _build_blocks_by_mode(
    windows: dataprep.DataWindows,
    mode: str,
    cfg: DataprepConfigs,
    logger: utils.Logger
):
    '''doc'''

    # mode difference
    if mode == 'train':
        label_fpath = cfg['train_lbl']
        valid_blks = cfg['train_valid_blks']
    else: # 'infer'
        label_fpath = None
        valid_blks = None

    builder_config=dataprep.BuilderConfig(
        image_fpath=cfg[f'{mode}_img'],
        label_fpath=label_fpath,
        config_fpath=cfg['input_config'],
        blks_dpath=cfg[f'{mode}_blks_dir'],
        all_blocks=cfg[f'{mode}_all_blks'],
        valid_blks=valid_blks
    )
    builder = dataprep.BlockCacheBuilder(windows, builder_config, logger)
    builder.build_block_cache()
    builder.build_valid_block_index(cfg[f'{mode}_px_thres'], rebuild=True)
