'''
Data block preparation pipeline.

Typical entry point: `build_data_cache()` called during data preparation.
This orchestrates layout creation, cache checking, block creation, and
block validation (for training).

For the complete configuration schema required to initialize and run the
pipeline, see `BlockCache.__init__`.
'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
import os
import typing
import warnings
import zipfile
import zlib
# third-party imports
import numpy
# local imports
import alias
import dataset
import utils

# --------------------------------Public  Class--------------------------------
class BlockCache:
    '''
    Wrapper class to streamline block cache creation and validation.

    Pipeline (in order):
    - `_get_layout()`: tile input raster(s) into windows based on config.
    - `_check_cache()`: find/remove missing/corrupted `.npz` block files.
    - `_create_cache()`: create `.npz` block files for tiles.
    - `_validate_cache()`: (training only) select valid blocks by config.

    Use `process(...)` to run the pipeline with flags for overwriting,
    cleaning, and validation control.

    Configuration is supplied via `_BlockCacheConfig` and specified in
    `__init__`.
    '''

    def __init__(
        self,
        config: _BlockCacheConfig,
        logger: utils.Logger,
    ):
        '''
        Initialize the class.

        Args:
            config: Composite configuration required to run the pipeline
                with following components (see corresponding dataclass
                docstrings for field-level details)
                - data : (`_DataPaths`)
                Input raster paths and per-block metadata configuration.
                - output : (`_OutputPaths`)
                Cache and artifact destinations.
                - block : (`_BlockConfig`)
                Block size and overlap parameters.
                - thres : (`_ValidThresholds` | `None`)
                Validation thresholds (required only for training).

            logger: Class-level logger used for all pipeline messages.
        '''

        # parse arguments
        self.cfg = config
        self.logger = logger

        # init attributes
        self.layout_dict: dict[str, alias.RasterWindow] = {}
        self.layout_meta: dict[str, typing.Any] = {}
        self.valid_blks: dict[str, str] = {}

    def process(
        self,
        *,
        overwrite_layout: bool = False,
        check_npz_files: bool = True,
        overwrite_cache: bool = False,
        validate_cache: bool = True
    ) -> 'BlockCache':
        '''
        Public API for pipeline execution with processing flags.

        Note: for inference blocks, all square blocks from the layout
        are considered valid thus validation step is skipped.
        '''

        # run sequence with flags
        self._get_layout(overwrite=overwrite_layout)
        self._check_cache(skip=check_npz_files)
        self._create_cache(overwrite=overwrite_cache)
        self._validate_cache(overwrite=validate_cache)
        return self # for possible chaining

    def _get_layout(self, overwrite: bool):
        '''Check cache component: layout.'''

        layout_dict = self.cfg.output.layout_dict
        layout_meta = self.cfg.output.layout_meta
        image_fpath = self.cfg.data.image_fpath
        label_fpath = self.cfg.data.label_fpath
        # get from cache
        if os.path.exists(layout_dict) and not overwrite:
            self.layout_dict = utils.load_pickle(layout_dict)
            self.layout_meta = utils.load_pickle(layout_meta)
            self.logger.log('INFO', f'Existing layout from: {layout_dict}')
            self.logger.log('INFO', f'Layout meta from: {layout_meta}')
        # get layout from input raster(s)
        else:
            self.logger.log('INFO', f'Creating/rewriting layout: {layout_dict}')
            layout = dataset.BlockLayout(
                blk_size=self.cfg.block.blk_size,
                overlap=self.cfg.block.overlap,
                logger=self.logger
            )
            layout.ingest(image_fpath, label_fpath)
            self.layout_dict = layout.blks
            self.layout_meta = layout.meta
            utils.write_pickle(layout_dict, layout.blks)
            utils.write_pickle(layout_meta, layout.meta)
            self.logger.log('INFO', f'New layout saved at: {layout_dict}')
            self.logger.log('INFO', f'Layout meta saved at: {layout_meta}')

    def _check_cache(self, skip: bool) -> None:
        '''Check block cache (.npz) files.'''

        # check files from expected list of .npz files
        if not skip:
            self.logger.log('INFO', 'Checking block .npz files')
            # parallel processing
            jobs = [(_valid_npz, (f,), {}) for f in self.square_blks.values()]
            results: list[dict[str, str]] = utils.ParallelExecutor().run(jobs)
            # parse results
            inv = [] # invalid blocks
            rm = [] # damaged files to be removed
            for result in results:
                fpath = result.get('invalid')
                if fpath:
                    inv.append(result['invalid'])
                if fpath and os.path.exists(fpath):
                    rm.append(result['invalid'])
            # remove corrupted/damaged files if present
            for fpath in rm:
                os.remove(fpath)
            # log checking results
            self.logger.log('INFO', f'Found {len(inv)} missing/damaged files')
            self.logger.log('INFO', f'Removed {len(rm)} damaged files')
        # skip checking
        else:
            self.logger.log('INFO', 'Skipping checking block files')

    def _create_cache(self, overwrite: bool) -> None:
        '''Create cached blocks as npz files.'''

        # self.layout needs to be initiated
        assert self.layout_dict

        # determine block files that need to be created
        todo: dict[str, alias.RasterWindow] = {}
        if overwrite:
            todo = self.layout_dict # all blocks in layout
        elif self.missing_blks:
            todo = {k: self.layout_dict[k] for k in self.missing_blks}
        else:
            todo = {}
        if not todo:
            self.logger.log('INFO', 'No data blocks to be created')
            return
        self.logger.log('INFO', f'{len(todo)} data blocks to be created')

        # prep block creation arguments
        img = self.cfg.data.image_fpath
        lbl = self.cfg.data.label_fpath
        cfg: _BlockConfig = utils.load_json(self.cfg.data.config_fpath)
        lkup = self.square_blks
        jobs = [(_do_a_blk, (b, img, lbl, cfg, lkup,), {}) for b in todo.items()]
        # parallel processing through all raster windows
        results = utils.ParallelExecutor().run(jobs)

        # log warning messages if any
        for msgs in results:
            if isinstance(msgs, list):
                for s in msgs:
                    self.logger.log('WARNING', f'{s}')

        # write an artifect: square block list
        utils.write_json(self.cfg.output.square_blks, self.square_blks)

    def _validate_cache(self, overwrite: bool) -> None:
        '''
        Get a list of file paths of the valid block with configuration.
        '''

        # self.layout needs to be initiated
        assert self.layout_meta

        # for inference blocks that need not validation. load all square blocks
        if not self.layout_meta.get('has_label'):
            _blks = self.cfg.output.square_blks
            self.logger.log('INFO', f'All square blocks from: {_blks} are valid')
            self.valid_blks = utils.load_json(self.cfg.output.square_blks)
            self.logger.log('INFO', f'Got {len(self.valid_blks)} blocks')
            return

        # for training blocks that need validation. try load valid blocks
        _blks = self.cfg.output.valid_blks
        # if already exists and not to overwrite
        if os.path.exists(_blks) and not overwrite:
            self.logger.log('INFO', f'Gather valid blocks from: {_blks}')
            self.valid_blks = utils.load_json(_blks)
            self.logger.log('INFO', f'Got {len(self.valid_blks)} blocks')
            return

        # otherwise create a new valid blocks json
        assert self.cfg.thres is not None # sanity check
        self.logger.log('INFO', 'Validating existing blocks')
        # prep block validation arguments
        val_px = self.cfg.thres.valid_px_ratio
        kw = {'water_px_ratio': self.cfg.thres.water_px_ratio}
        all_blks = self.square_blks
        jobs = [(_valid_block, (b, val_px,), kw) for b in all_blks.items()]
        # parallel processing through all blocks
        results: list[dict] = utils.ParallelExecutor().run(jobs)
        self.valid_blks = {
            r['valid']: all_blks[r['valid']] for r in results if 'valid' in r
        }
        # log and save
        self.logger.log('INFO', f'Got {len(self.valid_blks)} valid blocks')
        self.logger.log('INFO', f'Valid blocks json saved to {_blks}')
        utils.write_json(_blks, self.valid_blks)

    # -------------------------------properties-------------------------------
    @property
    def square_blks(self) -> dict[str, str]:
        '''Expected block (square) file list from layout.'''
        if self.layout_dict is not None:
            return {
                k: f'{self.cfg.output.blks_dpath}/{k}.npz'
                for k in self.layout_dict.keys()
            }
        raise ValueError('self.layout not initiated')

    @property
    def missing_blks(self) -> dict[str, str]:
        '''Blocks in `block_fpath_list` but in cache dir.'''
        existing = set(os.listdir(self.cfg.output.blks_dpath)) # filenames
        return{
            name: path for name, path in self.square_blks.items()
            if os.path.basename(path) not in existing # also filenames
        }

# ---------------------BlockCache related helper functions---------------------
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

def _do_a_blk(
    block: tuple[str, alias.RasterWindow],
    image_fpath: str,
    label_fpath: str | None,
    config: dataset.BlockMeta,
    fpath_lookup: dict[str, str]
) -> list[str]:
    '''Create new a block from input rasters (read by given Window).'''

    # parse arguments
    blk_name, blk_window = block

    # block meta takes options from general creation options
    meta = copy.deepcopy(config)
    # add general block meta
    meta['block_name'] = blk_name
    meta['block_shape'] = (blk_window.width, blk_window.height)
    assert blk_window.width == blk_window.height # sanity: square block

    # customize warnings context
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", category=RuntimeWarning)
        with utils.open_rasters(image_fpath, label_fpath) as (img, lbl):

            # read image array
            assert img is not None
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
    img: alias.RasterReader,
    window: alias.RasterWindow,
    dem_band: int,
    pad: int
) -> numpy.ndarray:
    '''Get a padded numpy array from image raster (windowed-read).'''

    # expand window within the original raster
    nw_x = max(window.col_off - pad, 0)
    nw_y = max(window.row_off - pad, 0)
    se_x = min(window.col_off + window.width + pad, img.width)
    se_y = min(window.row_off + window.height + pad, img.height)
    _window = alias.RasterWindow(nw_x, nw_y, se_x - nw_x, se_y - nw_y) # type: ignore

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

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _BlockCacheConfig:
    '''Composite configuration for `BlockCache`.'''
    data: _DataPaths
    output: _OutputPaths
    block: _BlockConfig
    thres: _ValidThresholds | None # none for inference blocks

@dataclasses.dataclass
class _DataPaths:
    '''Paths to raw inputs and per-block meta'''
    image_fpath: str            # path to raw image data (.tiff)
    label_fpath: str | None     # path to raw label data (.tiff)
    config_fpath: str           # path to raw metadata (.json)

@dataclasses.dataclass
class _OutputPaths:
    '''Destination paths for cache artifacts. Created if absent.'''
    blks_dpath: str     # dirpath to save block files
    layout_dict: str    # filepath (.pkl) to save the block layout
    layout_meta: str    # filepath (.json) to save the block layout metadata
    square_blks: str    # filepath (.json) to save dict of square block files
    valid_blks: str     # filepath (.json) to save a dict of valid block files

@dataclasses.dataclass
class _BlockConfig:
    '''Tilling parameters'''
    blk_size: int
    overlap: int

@dataclasses.dataclass
class _ValidThresholds:
    '''Validation thresholds for training mode.'''
    valid_px_ratio: float
    water_px_ratio: float

# -------------------------------Public Function-------------------------------
def build_data_cache(
    dataset_name: str,
    input_config: alias.ConfigType,
    cache_config: alias.ConfigType,
    logger: utils.Logger,
    mode: str,
) -> None:
    '''
    Build block cache end-to-end for data preparation.

    Args:
        dataset_name: Dataset ID for locating the inputs and output root.
        input_config: Mapping for raw dataset assets and shared options.
            Must provide
            - path to training image and label rasters
            - (optionally) path to inference image raster
            - path to per-block metadata JSON
        cache_config : Mapping for controlling tiling/layout, flags, and
            (training) filters. Must provide
            - block specs (size, overlap)
            - artifact names (layout_dict, layout_meta, square blocks,
                valid blocks)
            - flags (overwrite_layout, clean_npz, overwrite_cache,
                validate_cache)
            - filters (valid_px_thres, water_px_thres)

        logger: Root logger; a child logger 'cache' is used internally.
        mode: Determines whether validation runs and whether labels are
            required (`'training'` or `'inference'`).
    '''

    # get a child logger
    _logger = logger.get_child('cache')

    # make root path if not exist
    cache_dir = f'./data/{dataset_name}/cache'
    os.makedirs(cache_dir, exist_ok=True)

    # build by mode
    # get config for RasterBlockCache
    _logger.log('INFO', f'Building {mode} block cache')
    cache_cfg = _get_config(
        mode=mode,
        cache_dpath=cache_dir,
        dataset_name=dataset_name,
        input_config=input_config,
        cache_config=cache_config
    )
    # get creation options
    options = _get_creation_options(
        mode=mode,
        cache_config=cache_config
    )
    # process
    _ = BlockCache(cache_cfg, _logger).process(**options)
    _logger.log('INFO', f'Building {mode} block cache - completed')
    _logger.log_sep()

# ------------------------------private  function------------------------------
def _get_config(
    mode: str,
    cache_dpath: str,
    dataset_name: str,
    input_config: alias.ConfigType,
    cache_config: alias.ConfigType,
) -> _BlockCacheConfig:
    '''Helper to create `BlockCacheConfig` from config mappings.'''

    # cache root dir
    _dir = f'{cache_dpath}/{mode}'

    # accessors
    input_cfg = utils.ConfigAccess(input_config)
    cache_cfg = utils.ConfigAccess(cache_config)

    # compose cache config conomponents
    # data paths
    if mode == 'training':
        data_cfg = _DataPaths(
            image_fpath=input_cfg.get_asset(mode, 'images', dataset_name),
            label_fpath=input_cfg.get_asset(mode, 'labels', dataset_name),
            config_fpath=input_cfg.get_option('config')
        )
    else: # inference
        data_cfg = _DataPaths(
            image_fpath=input_cfg.get_asset(mode, 'images', dataset_name),
            label_fpath=None,
            config_fpath=input_cfg.get_option('config')
        )

    # output paths
    output_cfg = _OutputPaths(
        f'{_dir}/blocks',
        f'{_dir}/{cache_cfg.get_asset('artifacts', 'blocks', 'layout_dict')}',
        f'{_dir}/{cache_cfg.get_asset('artifacts', 'blocks', 'layout_meta')}',
        f'{_dir}/{cache_cfg.get_asset('artifacts', 'blocks', 'square')}',
        f'{_dir}/{cache_cfg.get_asset('artifacts', 'blocks', 'valid')}'
    )

    # block config
    block_cfg = _BlockConfig(
        cache_cfg.get_option('blocks', 'size'),
        cache_cfg.get_option('blocks','overlap')
    )

    # validation thresholds
    thres = _ValidThresholds(
        cache_cfg.get_option('filters', 'valid_px_thres'),
        cache_cfg.get_option('filters', 'water_px_thres')
    )
    if mode == 'inference':
        thres = None # no validation for inference blocks

    # make dirs if not exist
    os.makedirs(_dir, exist_ok=True)
    os.makedirs(f'{_dir}/blocks', exist_ok=True)

    # return composite config
    return _BlockCacheConfig(data_cfg, output_cfg, block_cfg, thres)

def _get_creation_options(
    mode: str,
    cache_config: alias.ConfigType
) -> dict[str, bool]:
    '''Helper to get creation options from config mappings.'''

    # accessors alisa
    _get_option = utils.ConfigAccess(cache_config).get_option

    if mode == 'training':
        return {
            'overwrite_layout': _get_option('flags', 'overwrite_layout'),
            'check_npz_files': _get_option('flags', 'clean_npz'),
            'overwrite_cache': _get_option('flags', 'overwrite_cache'),
            'validate_cache': _get_option('flags', 'validate_cache')
        }
    return {
        'overwrite_layout': _get_option('flags', 'overwrite_layout'),
        'check_npz_files': _get_option('flags', 'clean_npz'),
        'overwrite_cache': _get_option('flags', 'overwrite_cache'),
        'validate_cache': False # no validation for inference blocks
    }
