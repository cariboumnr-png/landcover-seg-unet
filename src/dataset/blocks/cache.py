'''Data blocks preperation pipeline.'''

# standard imports
import copy
import dataclasses
import os
import typing
import warnings
import zipfile
import zlib
# third-party imports
import numpy
import rasterio
import rasterio.io
import rasterio.windows
# local imports
import dataset
import dataset.blocks
import utils

# typing aliases
DatasetReader: typing.TypeAlias = rasterio.io.DatasetReader
Window: typing.TypeAlias = rasterio.windows.Window

# collection of cache file paths
@dataclasses.dataclass
class CachePaths:
    '''Doc'''
    image_fpath: str    # path to raw data
    label_fpath: str    # path to raw data
    meta_fpath: str     # path to raw data
    blks_dpath: str     # path to save block files (.npz)
    blk_scheme: str     # path to save block tiling scheme (.pkl)
    valid_blks: str     # paths of a list of valid block files (.pkl)

# module config
@dataclasses.dataclass
class CacheConfig:
    '''doc'''
    blk_size: int
    overlap: int
    valid_px_threshold: float
    water_px_threshold: float

# -----------------------------------Tilling-----------------------------------
def tile_rasters(
        paths: CachePaths,
        config: CacheConfig,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> dict[str, Window]:
    '''
    Prepare blocks from the input raster(s).
    '''

    # get a child logger
    logger=logger.get_child('tiler')

    # read from existing file
    if os.path.exists(paths.blk_scheme) and not overwrite:
        logger.log('INFO', f'Use existing block scheme: {paths.blk_scheme}')
        tiles = utils.load_pickle(paths.blk_scheme)
    # otherwise create if scheme not already exsited
    else:
        logger.log('INFO', f'Creating/re-writing scheme: {paths.blk_scheme}')
        processed = dataset.blocks.RasterBlockLayout(
            blk_size=config.blk_size,
            overlap=config.overlap,
            logger=logger
        )
        processed.ingest(paths.image_fpath, paths.label_fpath)
        tiles = processed.blks
        logger.log('INFO', f'New block scheme save at: {paths.blk_scheme}')
        utils.write_pickle(paths.blk_scheme, tiles)

    # return
    return tiles

# -----------------------------------Caching-----------------------------------
def create_block_cache(
        paths:CachePaths,
        *,
        logger: utils.Logger,
        run_cleanup: bool,
        overwrite: bool
    ) -> None:
    '''Create cached blocks as npz files.'''

    # get a child logger
    logger=logger.get_child('cache')

    # if chosen not to clean
    if run_cleanup:
        logger.log('INFO', f'Checking block .npz files at {paths.blks_dpath}')
        n_removed = _clean_up_bad_npz(paths.blks_dpath)
        logger.log('INFO', f'Found and removed {n_removed} bad .npz files')
    else:
        logger.log('INFO', 'Skipping checking for bad block .npz files')

    # load block scheme
    block_scheme = utils.load_pickle(paths.blk_scheme)
    # determine blocks to be processed by overwrite flag and file existence
    windows: dict[str, Window] = {}
    if overwrite:
        windows = block_scheme
    else:
        for key in block_scheme.keys():
            if not os.path.exists(f'{paths.blks_dpath}/block_{key}.npz'):
                windows.update({key: block_scheme[key]})
    if not windows:
        logger.log('INFO', 'No data blocks to be created')
        return
    logger.log('INFO', f'{len(windows)} data blocks to be created')

    # read raster and create blocks
    logger.log('INFO', 'Generating block cache files from input rasters')
    meta = utils.load_json(paths.meta_fpath)

    # parallel processing through all raster windows
    jobs = [(_block_from_ras, (paths, meta, w,), {})for w in windows.items()]
    results = utils.ParallelExecutor().run(jobs)

    # log warning messages if any
    for msgs in results:
        if isinstance(msgs, list):
            for s in msgs:
                logger.log('WARNING', f'{s}')

def _clean_up_bad_npz(blks_dpath: str) -> int:
    '''Simple helper to check and remove bad .npz files at dir.'''

    # parallel processing the blocks
    fpaths = utils.get_fpaths_from_dir(blks_dpath, '.npz')
    jobs = [(__npz_file_check, (fpath,), {}) for fpath in fpaths]
    results: list[dict[str, str]] = utils.ParallelExecutor().run(jobs)

    # parse results
    to_remove = []
    for result in results:
        if result.get('remove'):
            to_remove.append(result['remove'])

    # remove files and return count
    for fpath in to_remove:
        os.remove(fpath)
    return len(to_remove)

def __npz_file_check(blk_fpath: str) -> dict[str, str]:
    '''Check if a .npz block file is corrupted.'''
    # pass if the npz file can be loaded properly
    try:
        rb = dataset.DataBlock()
        rb.load_from_npz(blk_fpath)
        return {'pass': blk_fpath}
    # corrupted/damaged npz file to be removed
    except (zipfile.error, zlib.error):
        return {'remove': blk_fpath}

def _block_from_ras(
        paths: CachePaths,
        meta: dict[str, typing.Any],
        block: tuple[str, Window],
    ) -> list[str]:
    '''Create new a block from input rasters (read by given Window).'''

    # deep copy a meata dict to avoid cross-contanimation
    meta = copy.deepcopy(meta)

    # customize warnings context
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", category=RuntimeWarning)
        with rasterio.open(paths.label_fpath) as lbl, \
            rasterio.open(paths.image_fpath) as img:

            # add entries to meta
            meta['label_nodata'] = lbl.nodata
            meta['image_nodata'] = img.nodata
            meta['block_name'] = block[0]
            meta['block_shape'] = [block[1].width, block[1].height]

            # read original label and image arrays
            lbl_arr = lbl.read(window=block[1]) # (1, 256, 256)
            img_arr = img.read(window=block[1]) # (7, 256, 256)

            # get padded dem array
            dem_band = meta['band_assignment']['dem'] + 1 # rasterio is 1-based
            dem_pad = meta['dem_pad']
            dem_padded = _get_padded_dem(img, block[1], dem_band, dem_pad)

            # init and populate RasterBlock, meta as the kwargs
            raster_block = dataset.DataBlock()
            raster_block.create_from_rasters(img_arr, lbl_arr, dem_padded, meta)
            # write to target npz file
            raster_block.save_npz(f'{paths.blks_dpath}/block_{block[0]}.npz')

        # captures runtime warning if any
        return [
            warnings.formatwarning(
                message=wm.message,
                category=wm.category,
                filename=wm.filename,
                lineno=wm.lineno,
                line=wm.line
            ).rstrip()
            for wm in captured_warnings
        ]

def _get_padded_dem(
        ras: DatasetReader,
        window: Window,
        target_band: int,
        pad: int
    ) -> numpy.ndarray:
    '''Get a padded rasterio dataset for topographical channels.'''

    # expand window within the original raster
    nw_x = max(window.col_off - pad, 0)
    nw_y = max(window.row_off - pad, 0)
    se_x = min(window.col_off + window.width + pad, ras.width)
    se_y = min(window.row_off + window.height + pad, ras.height)
    new_window = Window(nw_x, nw_y, se_x - nw_x, se_y - nw_y) # type: ignore

    # get expanded array using the new window
    expanded = ras.read(target_band, window=new_window)

    # get required padding on each side
    pad_top = max(0, pad - window.row_off)
    pad_left = max(0, pad - window.col_off)
    pad_bottom = max(0, (window.row_off + window.height + pad) - ras.height)
    pad_right = max(0, (window.col_off + window.width + pad) - ras.width)

    # pad the expanded arr accordingly controlled by pad_width
    expanded_padded = numpy.pad(
        array=expanded,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='reflect'
    )

    # return
    return expanded_padded

# ---------------------------------Validating---------------------------------
def validate_blocks_cache(
        paths: CachePaths,
        config: CacheConfig,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> list[str]:
    '''Get a list of file paths of the valid block with given thres.'''

    # get a child logger
    logger=logger.get_child('valbk')

    # read and exit if list already pickled to file and not overwrite
    if os.path.exists(paths.valid_blks) and not overwrite:
        logger.log('INFO', f'Gathering valid blocks from: {paths.valid_blks}')
        v_fpaths = utils.load_pickle(paths.valid_blks)
        logger.log('INFO', f'Fetched {len(v_fpaths)} valid blocks')
        return v_fpaths

    # otherwise create a new list
    logger.log('INFO', f'Validating blocks at {paths.blks_dpath}')
    block_fpaths = utils.get_fpaths_from_dir(paths.blks_dpath, '.npz')
    jobs = [(_is_valid_block, (f, config,), {}) for f in block_fpaths]
    rs: list[dict] = utils.ParallelExecutor().run(jobs)
    v_fpaths = [r.get('valid', 0) for r in rs if r.get('valid', 0)]

    # save and return
    logger.log('INFO', f'Gathered {len(v_fpaths)} valid blocks')
    logger.log('INFO', f'List file saved to {paths.valid_blks}')
    utils.write_pickle(paths.valid_blks, v_fpaths)
    return v_fpaths

def _is_valid_block(
        block_fpath: str,
        config: CacheConfig
    ) -> dict[str, str]:
    '''Helper to flag whether a block is valid for downstream apps.'''

    # get meta from block
    meta = dataset.DataBlock().load_from_npz(block_fpath).meta

    # keep only square blocks
    if meta['block_shape'] != [config.blk_size, config.blk_size]:
        return {'invalid': block_fpath}
    # valid pixel ratio threshold
    if meta['valid_pixel_ratio']['block'] < config.valid_px_threshold:
        return {'invalid': block_fpath}
    # water pixel ratio threshold - do the calc here for now
    wat_idx = next(
        (int(k) for k, v in meta['label1_reclass_name'].items() if v == 'water')
    )
    wat_ratio = meta['label_count']['layer1'][wat_idx - 1] / \
        sum(meta['label_count']['layer1'])
    if wat_ratio > config.water_px_threshold:
        return {'invalid': block_fpath}
    # if all checks passed
    return {'valid': block_fpath}
