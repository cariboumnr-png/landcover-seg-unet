# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Block-building utilities that assemble dataset blocks from raster windows,
including single-block sampling for tests and cache/index generation.

Public APIs:
    - build_blocks: Generate block caches and valid-block indices.
    - build_a_block: Produce a single valid block using fit windows.'
'''

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
) -> None:
    '''
    Build cached blocks and a valid-block index for the selected mode.

    Args:
        mode: Operation mode, e.g., "fit" or "test".
        config: Block-building configuration with paths and thresholds.
        logger: Logger for status and diagnostics.
        rebuild: If True, recompute and overwrite existing indices.
    '''

    # mode derivatives
    assert mode in ['fit', 'test'], f'Invalid build mode: {mode}'
    windows_key = f'{mode}_windows'
    thres_key = f'blk_thres_{mode}'
    # get raster windows
    windows = utils.load_pickle(config[windows_key])
    # get a builder instance
    builder = _get_blocks_builder(windows, mode, config, logger)
    # build blocks
    builder.build_block_cache()
    builder.build_valid_block_index(config[thres_key], rebuild=rebuild)

def build_one_block(
    config: dataprep.BlockBuildingConfig,
    logger: utils.Logger,
    *,
    valid_px_per: float = 0.8,
    monitor_head: str = 'layer1',
    need_all_classes: bool = True
) -> blockbuilder.DataBlock:
    '''
    Construct and return one valid data block from fit windows.

    Args:
        config: Block-building configuration with input resources.
        logger: Logger for progress and selection details.
        valid_px_per: Minimum fraction of valid pixels required for the
            block. Defaults to `0.8`.
        monitor_head: Label head name used to check for class coverage.
            Defaults to `"layer1"`.
        need_all_classes: If True, require all classes to be present
            under the monitor head; if False, skip the class-coverage
            check. Defaults to `True`.

    Returns:
        DataBlock: A block meeting the validity threshold, and class
            coverage criterion if enabled.
    '''

    # get fit raster windows
    windows: mapper.DataWindows = utils.load_pickle(config['fit_windows'])
    # get a builder instance
    builder = _get_blocks_builder(windows, 'fit', config, logger)
    # get a deterministic coordinate sequence to iterate
    coords = list(windows.image_windows.keys()) # from image windows
    random.Random(42).shuffle(coords)
    # build a valid block and return
    i = 0
    while True:
        print('Searching for a valid raster window...', end='\r', flush=True)
        try:
            block = builder.single_block(coords[i])
        except ValueError: # likely an empty window for the rasters
            i += 1
            continue
        meta = block.meta
        data = block.data
        if meta['valid_pixel_ratio']['block'] >= valid_px_per and \
            (all(meta['label_count'][monitor_head]) or not need_all_classes):
            logger.log('INFO', f'Fetched a valid block at coord: {coords[i]}')
            logger.log('DEBUG', 'Criteria:')
            logger.log('DEBUG', f'Minimum valid pixel: {valid_px_per:.2f}')
            logger.log('DEBUG', f'Focused head: {monitor_head}')
            logger.log('DEBUG', f'Requires all classes: {need_all_classes}')
            # normalize
            data.image_normalized = numpy.empty_like(data.image)
            for i, arr in enumerate(data.image):
                std_safe = numpy.where(numpy.std(arr) == 0, 1, numpy.std(arr))
                norm = (arr - numpy.mean(arr)) / std_safe   # (H, W)
                data.image_normalized[i] = norm       # (C, H, W)
            # return the block instance
            return block
        i += 1

def _get_blocks_builder(
    windows: mapper.DataWindows,
    mode: str,
    config: dataprep.IOConfig,
    logger: utils.Logger
) -> blockbuilder.BlockCacheBuilder:
    '''Return a block builder configured for the given mode.'''

    # gather configurations by mode
    if mode == 'fit':
        image_fpath = config['fit_input_img']
        label_fpath = config['fit_input_lbl']
        config_fpath = config['input_config']
        blks_dpath = config['fit_blks_dir']
        all_blocks = config['fit_all_blks']
        valid_blks = config['fit_valid_blks']
    elif mode == 'test':
        assert config['test_input_img'] # sanity type check
        image_fpath = config['test_input_img']
        label_fpath = config['test_input_lbl']
        config_fpath = config['input_config']
        blks_dpath = config['test_blks_dir']
        all_blocks = config['test_all_blks']
        valid_blks = config['test_valid_blks']
    else:
        raise ValueError(f'Invalid builder mode {mode}')

    # get a builder configuration dataclass
    builder_config=blockbuilder.BuilderConfig(
        image_fpath=image_fpath,
        label_fpath=label_fpath,
        config_fpath=config_fpath,
        blks_dpath=blks_dpath,
        all_blocks=all_blocks,
        valid_blks=valid_blks
    )

    # build and return a builder instance
    return blockbuilder.BlockCacheBuilder(windows, builder_config, logger)
