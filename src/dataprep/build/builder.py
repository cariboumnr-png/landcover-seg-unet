'''doc'''

# local imports
import alias
import dataprep
import grid
import utils

def build_cache(
    mode: str,
    world_grid: grid.GridLayout,
    data_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
    logger: utils.Logger
) -> None:
    '''doc'''

    # check mode
    if mode not in ['training', 'inference']:
        raise ValueError(f'Invalid mode: {mode}: "training" or "inference"')

    # get a child logger
    logger = logger.get_child('dprep')

    # config accessors
    data_cfg = utils.ConfigAccess(data_config)
    artifact_cfg = utils.ConfigAccess(artifact_config)

    # get file paths depending on mode
    image_fpath = data_cfg.get_asset(mode, 'images', 'demo')
    if mode == 'training':
        label_fpath = data_cfg.get_asset(mode, 'labels', 'demo')
    else:
        label_fpath = None

    # map data to world grid
    img_windows, lbl_windows = dataprep.map_rasters(
        world_grid=world_grid,
        image_fpath=image_fpath,
        label_fpath=label_fpath,
        logger=logger
    )
    # build data blocks
    pipeline = dataprep.BlockCachePipeline(
        windows=dataprep.Windows(
            image_windows=img_windows,
            label_windows=lbl_windows,
            expected_shape=world_grid.tile_size
        ),
        inputs=dataprep.DataPaths(
            image_fpath=image_fpath,
            label_fpath=label_fpath,
            config_fpath=data_cfg.get_option('config'),
        ),
        output=dataprep.Artifacts(
            blks_dpath=artifact_cfg.get_option(f'{mode}_blocks_dir'),
            all_blocks=artifact_cfg.get_option(f'{mode}_all_blocks'),
            valid_blks=artifact_cfg.get_option(f'{mode}_valid_blocks')
        ),
        logger=logger
    )
    pipeline.build_block_cache()
    if mode == 'training':
        pipeline.build_valid_block_index(px_thres=0.75, rebuild=True)
    else:
        pipeline.build_valid_block_index(px_thres=0.0, rebuild=True)
