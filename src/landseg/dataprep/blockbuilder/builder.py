'''doc'''

# standard imports
import random
# third-party imports
import numpy
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
    rebuild: bool = False,
) -> blockbuilder.DataBlock | None:
    '''doc'''

    # mode derivatives
    windows_key = f'{mode}_windows'
    thres_key = f'blk_thres_{mode}'
    # get raster windows
    windows = utils.load_pickle(config[windows_key])
    # get a builder instance
    builder = _get_blocks_builder(windows, mode, config, logger)
    # build blocks
    builder.build_block_cache()
    builder.build_valid_block_index(config[thres_key], rebuild=rebuild)

def build_a_block(
    config: dataprep.BlockBuildingConfig,
    logger: utils.Logger
) -> blockbuilder.DataBlock:
    '''doc'''

    # get fit raster windows
    windows: mapper.DataWindows = utils.load_pickle(config['fit_windows'])
    # get a builder instance
    builder = _get_blocks_builder(windows, 'fit', config, logger)
    # get a deterministic coordinate sequence to iterate
    coords = list(windows.image_windows.keys()) # from image windows
    random.Random(42).shuffle(coords)
    # build a block (>= 80% valid pixels) and return
    i = 0
    while True:
        print('Searching for a good raster window...', end='\r', flush=True)
        try:
            block = builder.build_a_block(coords[i])
        except ValueError: # likely an empty window for the rasters
            i += 1
            continue
        if block.meta['valid_pixel_ratio']['block'] >= 0.8:
            logger.log('INFO', f'Fetched a valid block at coord: {coords[i]}')
            # normalize
            block.data.image_normalized = numpy.empty_like(block.data.image)
            for i, arr in enumerate(block.data.image):
                std = numpy.std(arr)
                std_safe = numpy.where(std == 0, 1, std)
                norm = (arr - numpy.mean(arr)) / std_safe   # (H, W)
                block.data.image_normalized[i] = norm       # (C, H, W)
            # return the block instance
            return block
        i += 1

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
        label_fpath=config['test_input_lbl']
        config_fpath=config['input_config']
        blks_dpath=config['test_blks_dir']
        all_blocks=config['test_all_blks']
        valid_blks=config['test_valid_blks']
    else:
        raise ValueError(f'Invalid builder mode {mode}')

    # get a builder and return
    builder_config=blockbuilder.BuilderConfig(
        image_fpath = image_fpath,
        label_fpath = label_fpath,
        config_fpath = config_fpath,
        blks_dpath = blks_dpath,
        all_blocks = all_blocks,
        valid_blks=valid_blks
    )
    builder = blockbuilder.BlockCacheBuilder(windows, builder_config, logger)
    return builder
