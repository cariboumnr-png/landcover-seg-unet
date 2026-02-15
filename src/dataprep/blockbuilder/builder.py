'''doc'''

# local import
import dataprep
import utils

def get_block_builder(
    windows: dataprep.DataWindows,
    mode: str,
    config: dataprep.DataprepConfigs,
    logger: utils.Logger
) -> dataprep.BlockCacheBuilder:
    '''doc'''

    # mode difference
    if mode == 'train':
        image_fpath=config['train_img']
        label_fpath=config['train_lbl']
        config_fpath=config['input_config']
        blks_dpath=config['train_blks_dir']
        all_blocks=config['train_all_blks']
        valid_blks=config['train_valid_blks']
    elif mode == 'infer':
        image_fpath=config['infer_img']
        label_fpath=None
        config_fpath=config['input_config']
        blks_dpath=config['infer_blks_dir']
        all_blocks=config['infer_all_blks']
        valid_blks=None

    else:
        raise ValueError(f'Invalid builder mode {mode}')

    assert image_fpath # sanity type check
    # get a builder and return
    builder_config=dataprep.BuilderConfig(
        image_fpath = image_fpath,
        label_fpath = label_fpath,
        config_fpath = config_fpath,
        blks_dpath = blks_dpath,
        all_blocks = all_blocks,
        valid_blks=valid_blks
    )
    builder = dataprep.BlockCacheBuilder(windows, builder_config, logger)
    return builder
