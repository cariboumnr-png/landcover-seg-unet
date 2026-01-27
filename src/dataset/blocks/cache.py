'''Data blocks preperation pipeline.'''

from __future__ import annotations
# standard imports
import contextlib
import copy
import dataclasses
import os
import typing
import warnings
import zipfile
import zlib
# third-party imports
import numpy
import omegaconf
import rasterio
import rasterio.io
import rasterio.windows
# local imports
import dataset
import dataset.blocks
import utils

# typing aliases
# from rasterio
DatasetReader: typing.TypeAlias = rasterio.io.DatasetReader
Window: typing.TypeAlias = rasterio.windows.Window
# from local
BlockLayout: typing.TypeAlias = dataset.blocks.RasterBlockLayout
BlockMetaDict: typing.TypeAlias = dataset.blocks.BlockMetaDict

class RasterBlockCache:
    '''doc'''

    def __init__(
            self,
            config: BlockCacheConfig,
            logger: utils.Logger,
        ):
        '''doc'''

        # parse arguments
        self.cfg = config
        self.logger = logger.get_child('cache') # get a child logger

        # init attributes
        self.layout: BlockLayout | None = None
        self.valid_blks: dict[str, str] = {}

    def process(self, **kwargs) -> None:
        '''doc'''
        # run sequence
        self.get_layout(overwrite=kwargs.get('overwrite_layout', False))
        self.check_cache(skip=kwargs.get('check_npz_files', False))
        self.create_cache(overwrite=kwargs.get('overwrite_cache', False))
        self.validate_cache(overwrite=kwargs.get('validate_cache', False))

    def get_layout(self, overwrite: bool):
        '''Check cache component: layout.'''

        layout_fpath = self.cfg.output.blk_layout
        image_fpath = self.cfg.data.image_fpath
        label_fpath = self.cfg.data.label_fpath
        # get from cache
        if os.path.exists(layout_fpath) and not overwrite:
            self.logger.log('INFO', f'Use existing block scheme: {layout_fpath}')
            self.layout = utils.load_pickle(layout_fpath)
        # get layout from input raster(s)
        else:
            self.logger.log('INFO', f'Creating/re-writing scheme: {layout_fpath}')
            self.layout = BlockLayout(
                blk_size=self.cfg.block.blk_size,
                overlap=self.cfg.block.overlap,
                logger=self.logger
            )
            self.layout.ingest(image_fpath, label_fpath)
            self.logger.log('INFO', f'New block scheme save at: {layout_fpath}')
            utils.write_pickle(layout_fpath, self.layout)

    def check_cache(self, skip: bool) -> None:
        '''Check block cache (.npz) files.'''

        # check files from expected list of .npz files
        if not skip:
            self.logger.log('INFO', 'Checking block .npz files')
            # parallel processing
            jobs = [(_valid_npz, (f,), {}) for f in self.blk_fpath_dict.values()]
            results: list[dict[str, str]] = utils.ParallelExecutor().run(jobs)
            # parse results
            invalid = []
            to_remove = []
            for result in results:
                fpath = result.get('invalid')
                if fpath:
                    invalid.append(result['invalid'])
                if fpath and os.path.exists(fpath):
                    to_remove.append(result['invalid'])
            # remove corrupted/damaged files if present
            for fpath in to_remove:
                os.remove(fpath)
            # log checking results
            self.logger.log('INFO', f'Found {len(invalid)} invalid block files')
            self.logger.log('INFO', f'{len(to_remove)} damaged files removed')
        # skip checking
        else:
            self.logger.log('INFO', 'Skipping checking block files')

    def create_cache(self, overwrite: bool) -> None:
        '''Create cached blocks as npz files.'''

        # self.layout needs to be initiated
        assert self.layout is not None

        # determine block files that need to be created
        todo: dict[str, Window] = {}
        if overwrite:
            todo = self.layout.blks # all blocks in layout
        elif self.missing_blk_files:
            todo = {k: self.layout.blks[k] for k in self.missing_blk_files}
        else:
            todo = {}
        if not todo:
            self.logger.log('INFO', 'No data blocks to be created')
            return
        self.logger.log('INFO', f'{len(todo)} data blocks to be created')

        # prep block creation arguments
        img = self.cfg.data.image_fpath
        lbl = self.cfg.data.label_fpath
        meta: BlockMetaDict = utils.load_json(self.cfg.data.meta_fpath)
        lookup = self.blk_fpath_dict
        jobs = [(_do_a_blk, (b, img, lbl, meta, lookup,), {}) for b in todo.items()]
        # parallel processing through all raster windows
        results = utils.ParallelExecutor().run(jobs)

        # log warning messages if any
        for msgs in results:
            if isinstance(msgs, list):
                for s in msgs:
                    self.logger.log('WARNING', f'{s}')

    def validate_cache(self, overwrite: bool) -> None:
        '''Get a list of file paths of the valid block with given thres.'''

        # output json fpath
        valid_blks = self.cfg.output.blks_valid

        # read and exit if list already pickled to file and not overwrite
        if os.path.exists(valid_blks) and not overwrite:
            self.logger.log('INFO', f'Gather valid blocks from: {valid_blks}')
            self.valid_blks = utils.load_json(valid_blks)
            self.logger.log('INFO', f'Got {len(self.valid_blks)} valid blocks')

        # otherwise create a new list
        self.logger.log('INFO', 'Validating existing blocks')
        # prep block validation arguments
        val_px = self.cfg.thres.valid_px_ratio
        kw = {'water_px_ratio': self.cfg.thres.water_px_ratio}
        jobs = [(_valid_block, (b, val_px,), kw) for b in self.blk_fpath_dict.items()]
        # parallel processing through all blocks
        results: list[dict] = utils.ParallelExecutor().run(jobs)
        self.valid_blks = {
            r['valid']: self.blk_fpath_dict[r['valid']]
            for r in results if 'valid' in r
        }

        # log and save
        self.logger.log('INFO', f'Gathered {len(self.valid_blks)} valid blocks')
        self.logger.log('INFO', f'List file saved to {valid_blks}')
        utils.write_json(valid_blks, self.valid_blks)

    # -------------------------------properties-------------------------------
    @property
    def blk_fpath_dict(self) -> dict[str, str]:
        '''Expected block file list from layout.'''
        if self.layout is not None:
            return {
                k: f'{self.cfg.output.blks_dpath}/{k}.npz'
                for k in self.layout.blks.keys()
            }
        raise ValueError('self.layout not initiated')

    @property
    def missing_blk_files(self) -> dict[str, str]:
        '''Blocks in `block_fpath_list` but in cache dir.'''
        existing = set(os.listdir(self.cfg.output.blks_dpath)) # filenames
        return{
            name: path for name, path in self.blk_fpath_dict.items()
            if os.path.basename(path) not in existing # also filenames
        }

# ------------------------------helper functions------------------------------
def _valid_npz(blk_fpath: str) -> dict[str, str]:
    '''Check if a .npz block file is corrupted.'''

    rb = dataset.DataBlock()
    # pass if the npz file can be loaded properly
    try:
        rb.load(blk_fpath)
        return {'pass': blk_fpath}
    # flag absent/corrupted/damaged npz file
    except (FileNotFoundError, zipfile.error, zlib.error):
        return {'invalid': blk_fpath}

@contextlib.contextmanager
def _open_rasters(
        image_fpath: str,
        label_fpath: str | None
    ) -> typing.Iterator[tuple[DatasetReader, DatasetReader | None]]:
    '''Return open raster context.'''

    # if both image and label are provided - typically for training
    if label_fpath is not None:
        with rasterio.open(image_fpath) as img, \
            rasterio.open(label_fpath) as lbl:
            yield img, lbl
    # else if only image is provided - this is for inference
    else:
        with rasterio.open(image_fpath) as img:
            yield img, None

def _do_a_blk(
        block: tuple[str, Window],
        image_fpath: str,
        label_fpath: str | None,
        meta: BlockMetaDict,
        fpath_lookup: dict[str, str]
    ) -> list[str]:
    '''Create new a block from input rasters (read by given Window).'''

    # parse arguments
    blk_name, blk_window = block

    # deep copy a meata dict to avoid cross-contanimation
    meta = copy.deepcopy(meta)
    # add general block meta
    meta['block_name'] = blk_name
    meta['block_shape'] = (blk_window.width, blk_window.height)
    assert blk_window.width == blk_window.height # sanity: square block

    # customize warnings context
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", category=RuntimeWarning)
        with _open_rasters(image_fpath, label_fpath) as (img, lbl):

            # read image array
            img_arr = img.read(window=blk_window)
            meta['image_nodata'] = img.nodata
            # get padded dem array from image
            padded_dem = _pad_dem(
                img=img,
                window=blk_window,
                dem_band=meta['band_map']['dem'],
                pad=meta['dem_pad']
            )

            # read label array if provided
            if lbl is not None:
                lbl_arr = lbl.read(window=block[1])
                meta['label_nodata'] = lbl.nodata
            else:
                lbl_arr = None

            # init and populate RasterBlock
            raster_block = dataset.DataBlock().create(
                img_arr=img_arr,
                lbl_arr=lbl_arr,
                padded_dem=padded_dem,
                meta=meta
            )
            # write to target npz file
            raster_block.save(fpath_lookup[blk_name])

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

def _pad_dem(
        img: DatasetReader,
        window: Window,
        dem_band: int,
        pad: int
    ) -> numpy.ndarray:
    '''Get a padded numpy array from image raster (windowed-read).'''

    # expand window within the original raster
    nw_x = max(window.col_off - pad, 0)
    nw_y = max(window.row_off - pad, 0)
    se_x = min(window.col_off + window.width + pad, img.width)
    se_y = min(window.row_off + window.height + pad, img.height)
    _window = Window(nw_x, nw_y, se_x - nw_x, se_y - nw_y) # type: ignore

    # get expanded array using the new window
    # band number in rasterio.read is 1-based
    expanded = img.read(dem_band + 1, window=_window)

    # get required padding on each side - no padding if within raster bound
    pad_top = max(0, pad - window.row_off)
    pad_left = max(0, pad - window.col_off)
    pad_bottm = max(0, (window.row_off + window.height + pad) - img.height)
    pad_right = max(0, (window.col_off + window.width + pad) - img.width)

    # pad the expanded arr accordingly controlled by pad_width and return
    expanded_padded = numpy.pad(
        array=expanded,
        pad_width=((pad_top, pad_bottm), (pad_left, pad_right)),
        mode='reflect'
    )
    return expanded_padded

def _valid_block(
        block: tuple[str, str],
        valid_px_threshold: float,
        *,
        water_px_ratio: float
    ) -> dict[str, str]:
    '''Helper to flag whether a block is valid for downstream apps.'''

    # parse arguments
    name, fpath = block
    # get meta from block
    meta = dataset.DataBlock().load(fpath).meta

    # valid pixel ratio threshold
    if meta['valid_pixel_ratio']['block'] < valid_px_threshold:
        return {'invalid': name}

    # if label is provided, threshold on ratio of certain labels from layer1
    if meta['has_label']:
        # assertion only for typing chek
        assert 'label_count' in meta
        assert 'label1_reclass_name' in meta
        lbl_count = meta['label_count']['layer1']
        cls_names = meta['label1_reclass_name']
        # currently implemented: threshold on water pixel ratio
        wat_idx = next((int(k) for k, v in cls_names.items() if v == 'water'))
        wat_ratio = meta['label_count']['layer1'][wat_idx - 1] / sum(lbl_count)
        if wat_ratio > water_px_ratio:
            return {'invalid': name}

    # if all checks passed
    return {'valid': name}

# -----------------------------config dataclasses-----------------------------
# composite config
@dataclasses.dataclass
class BlockCacheConfig:
    '''Config for `RasterBlockCache`.'''
    data: DataPaths
    output: OutputPaths
    block: BlockConfig
    thres: ValidThresholds

@dataclasses.dataclass
class DataPaths:
    '''Collection of file paths to raw data.'''
    image_fpath: str            # path to raw image data (.tiff)
    label_fpath: str | None     # path to raw label data (.tiff)
    meta_fpath: str             # path to raw metadata (.json)

@dataclasses.dataclass
class OutputPaths:
    '''Collection of cache file paths.'''
    blks_dpath: str     # dirpath to save block files
    blk_layout: str     # filepath (.pkl) to save the block layout
    blks_valid: str     # filepath (.json) to save a dict of valid block files

@dataclasses.dataclass
class BlockConfig:
    '''Block layout config.'''
    blk_size: int
    overlap: int

@dataclasses.dataclass
class ValidThresholds:
    '''Block validation thresholds.'''
    valid_px_ratio: float
    water_px_ratio: float

# ----------------------------factory-like function----------------------------
def build_data_cache(
        config: omegaconf.DictConfig,
        logger: utils.Logger,
    ) -> None:
    '''Create cache blocks from scratch'''

    # create cache dirs
    os.makedirs(config.paths.cache, exist_ok=True)
    os.makedirs(config.paths.blksdpath, exist_ok=True)

    # init cache config dataclass
    cache_config = BlockCacheConfig(
        # set input data paths
        data=DataPaths(
            image_fpath=config.inputs.image,
            label_fpath=config.inputs.label,
            meta_fpath=config.inputs.meta
        ),
        # set output paths
        output=OutputPaths(
            blks_dpath=config.paths.blksdpath,
            blk_layout=config.paths.blklayout,
            blks_valid=config.paths.blkvalid
        ),
        # set block ocnfig
        block=BlockConfig(
            blk_size=config.blocks.size,
            overlap=config.blocks.overlap
        ),
        # set validation thresholds
        thres=ValidThresholds(
            valid_px_ratio=config.filters.pxthres,
            water_px_ratio=config.filters.watthres
        )
    )

    # get execution options
    options: dict[str, bool] = {
        'overwrite_layout': config.overwrite.layout,
        'check_npz_files': config.flags.clean_npz,
        'overwrite_cache': config.overwrite.cache,
        'validate_cache': config.overwrite.valid
    }

    # process block caching procedures
    block_cache = dataset.blocks.RasterBlockCache(cache_config, logger)
    block_cache.process(**options)
