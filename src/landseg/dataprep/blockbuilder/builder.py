'''doc'''

# local imports
import landseg.dataprep as dataprep
import landseg.dataprep.blockbuilder as blockbuilder
import landseg.dataprep.mapper as mapper
import landseg.utils as utils

def build_blocks(
    mode: str,
    config: dataprep.BlockBuildingConfig,
    logger: utils.Logger,
    *,
    rebuild: bool = False
) -> None:
    '''doc'''

    # mode derivatives
    windows_key = f'{mode}_windows'
    thres_key = f'blk_thres_{mode}'
    # fit raster windows
    windows = utils.load_pickle(config[windows_key])
    # build fit blocks
    builder = _get_blocks_builder(windows, mode, config, logger)
    builder.build_block_cache()
    builder.build_valid_block_index(config[thres_key], rebuild=rebuild)

def _get_blocks_builder(
    windows: mapper.DataWindows,
    mode: str,
    config: dataprep.IOConfig,
    logger: utils.Logger
) -> blockbuilder.BlockCacheBuilder:
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
    builder = blockbuilder.BlockCacheBuilder(windows, builder_config, logger)
    return builder
