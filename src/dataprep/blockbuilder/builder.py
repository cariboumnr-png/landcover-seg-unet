'''doc'''

# local import
import dataprep
import utils

def get_block_builder(
    windows: dataprep.DataWindows,
    mode: str,
    config: dataprep.BlockBuilderConfigs,
    logger: utils.Logger
) -> dataprep.BlockCacheBuilder:
    '''doc'''

    # mode selection
    if mode == 'fit':
        image_fpath=config['fit_input_img']
        label_fpath=config['fit_input_lbl']
        config_fpath=config['input_config']
        blks_dpath=config['fit_blks_dir']
        all_blocks=config['fit_all_blks']
        valid_blks=config['fit_valid_blks']
    elif mode == 'test':
        assert config['test_input_img'] # sanity type check
        image_fpath=config['test_input_img']
        label_fpath=None
        config_fpath=config['input_config']
        blks_dpath=config['test_blks_dir']
        all_blocks=config['test_all_blks']
        valid_blks=None

    else:
        raise ValueError(f'Invalid builder mode {mode}')

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
