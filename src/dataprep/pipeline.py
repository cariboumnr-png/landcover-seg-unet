'''doc'''

# local imports
import alias
import dataprep
import grid
import utils

def prepare_data(
    world_grid: grid.GridLayout,
    data_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    cache_config: alias.ConfigType,
    logger: utils.Logger
):
    '''doc'''

    # config accessors
    data_cfg = utils.ConfigAccess(data_config)
    artifact_cfg = utils.ConfigAccess(artifact_config)
    cache_cfg = utils.ConfigAccess(cache_config)

    # build blocks from all provided rasters
    _build_datablocks(world_grid, data_cfg, artifact_cfg, cache_cfg, logger)

def _build_datablocks(
    world_grid: grid.GridLayout,
    data_cfg: utils.ConfigAccess,
    artifact_cfg: utils.ConfigAccess,
    cache_cfg: utils.ConfigAccess,
    logger: utils.Logger
) -> None:
    '''doc'''

    # get a child logger for this module
    logger = logger.get_child('dprep')

    # evaluate input data
    train_img = data_cfg.get_option('training', 'image', default=None)
    train_lbl = data_cfg.get_option('training', 'label', default=None)
    infer_img = data_cfg.get_option('inference', 'image', default=None)
    input_data_config = data_cfg.get_option('config', default=None)

    # sanity checks on required items
    assert train_img is not None
    assert input_data_config is not None

    # get validation thresholds
    train_px_thres = cache_cfg.get_option('threshold', 'train_blocks_px')
    infer_px_thres = cache_cfg.get_option('threshold', 'infer_blocks_px')

    # build blocks for training data
    # map rasters to grid
    windows = dataprep.map_rasters(world_grid, train_img, train_lbl, logger)
    # ger cache builder i/o config
    builder_config=dataprep.BuilderConfig(
        image_fpath=train_img,
        label_fpath=train_lbl,
        config_fpath=input_data_config,
        blks_dpath=artifact_cfg.get_option('training_blocks_dir'),
        all_blocks=artifact_cfg.get_option('training_all_blocks'),
        valid_blks=artifact_cfg.get_option('training_valid_blocks')
    )
    # create builder - build - validate
    cache_builder = dataprep.BlockCacheBuilder(windows, builder_config, logger)
    cache_builder.build_block_cache()
    cache_builder.build_valid_block_index(train_px_thres, rebuild=True)

    # build blocks for inference data if provided
    if not infer_img:
        return
    # map rasters to grid
    windows = dataprep.map_rasters(world_grid, infer_img, None, logger)
    # ger cache builder i/o config
    builder_config=dataprep.BuilderConfig(
        image_fpath=infer_img,
        label_fpath=None,
        config_fpath=input_data_config,
        blks_dpath=artifact_cfg.get_option('inference_blocks_dir'),
        all_blocks=artifact_cfg.get_option('inference_all_blocks'),
        valid_blks=artifact_cfg.get_option('inference_valid_blocks')
    )
    # create builder - build - validate
    cache_builder = dataprep.BlockCacheBuilder(windows, builder_config, logger)
    cache_builder.build_block_cache()
    cache_builder.build_valid_block_index(infer_px_thres, rebuild=True)
