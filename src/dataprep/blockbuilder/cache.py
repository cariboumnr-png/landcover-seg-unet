'''
Data block preparation pipeline.

This module consumes precomputed raster read windows and builds a block
cache on disk:

- verifies integrity of existing '.npz' block files,
- creates any missing block files (gap filling only),
- optionally builds an index of valid blocks for training.

Notes:
- Window tiling and world-grid construction are handled upstream.
- Integrity checking always runs.
- Block creation only fills gaps; it does not rebuild existing blocks.
- Class balance or content filtering is not performed here; handle that
  in training samplers or policies.
'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
import os
import zipfile
import zlib
# third-party imports
import numpy
# local imports
import alias
import dataprep
import utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BuilderConfig:
    '''I/O Paths to raw rasters and artifacts during block building.'''
    image_fpath: str                        # path to raw image data (.tiff)
    label_fpath: str | None                 # path to raw label data (.tiff)
    config_fpath: str                       # path to raw metadata (.json)
    blks_dpath: str         # dirpath to save block files
    all_blocks: str         # filepath (.json) to all blocks
    valid_blks: str | None  # filepath (.json) to valid blocks

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _BlockCreationContext:
    '''Immutable inputs required to create a single block.'''
    name: str
    meta: dataprep.BlockMeta
    img_path: str
    img_window: alias.RasterWindow
    lbl_path: str | None
    lbl_window: alias.RasterWindow | None
    npz_fpath: str

# --------------------------------Public  Class--------------------------------
class BlockCacheBuilder:
    '''
    Orchestrates block cache validation and creation from supplied read
    windows.

    Processing stages (in order):
    - _load_read_windows(): intersect image/label windows and drop any
      that do not match the expected shape.
    - _check_block_files(): always runs; flags missing/corrupted '.npz'
      and removes damaged files.
    - _create_block_files(): creates '.npz' only for missing blocks.
    - build_valid_block_index(): optional training step to construct or
      load the valid-block index.

    Window tiling must be provided upstream.
    '''

    def __init__(
        self,
        windows: dataprep.DataWindows,
        config: BuilderConfig,
        logger: utils.Logger,
    ):
        '''
        Initialize the pipeline.

        Args:
            windows: Image/label read windows and expected shape.
            inputs: Paths to rasters and block meta config.
            output: Paths for block cache and index artifacts.
            logger: Logger for progress and diagnostics.
        '''

        # parse arguments
        self.windows = windows
        self.config = config
        self.logger = logger

        # init attributes
        self.shared_coords: set[tuple[int, int]] = set()
        self.all_blocks: dict[str, str] = {}
        self.valid_blks: dict[str, str] = {}

        # make sure output dir for the blocks exist
        os.makedirs(self.config.blks_dpath, exist_ok=True)

    def build_block_cache(self) -> 'BlockCacheBuilder':
        '''
        Run the cache preparation sequence.

        Steps:
            1) Load and filter shared read windows by expected shape.
            2) Check existing '.npz' blocks; remove any damaged files.
            3) Create '.npz' blocks only where missing.

        Returns:
            Self, for optional chaining.

        Notes:
            Validation is not performed here. Use
            'build_valid_block_index()' for training scenarios.
        '''

        # run sequence with flags
        self._prepare_block_windows()
        self._validate_existing_blocks()
        self._create_missing_blocks()
        return self # for possible chaining

    def build_valid_block_index(
        self,
        px_thres: float,
        *,
        rebuild: bool = False
    ) -> None:
        '''
        Build or load the valid-block index for training.

        Args:
            px_thres: Threshold on block-level valid pixel ratio.
            rebuild: Force re-evaluation even if an index exists.

        Side effects:
            - Populates 'self.valid_blks'.
            - Writes the valid-blocks JSON to 'output.valid_blks'.

        Note: `px_thres` == 0.0 means no validation.
        '''

        # if px_thres == 0.0 then skip validation
        if not px_thres or self.config.valid_blks is None:
            _blks = self.config.all_blocks
            self.logger.log('INFO', f'All blocks from: {_blks} are valid')
            self.valid_blks = utils.load_json(self.config.all_blocks)
            self.logger.log('INFO', f'Got {len(self.valid_blks)} blocks')
            return

        # if already exists and not to overwrite
        _blks = self.config.valid_blks
        if os.path.exists(_blks) and not rebuild:
            self.logger.log('INFO', f'Gather valid blocks from: {_blks}')
            self.valid_blks = utils.load_json(_blks)
            self.logger.log('INFO', f'Got {len(self.valid_blks)} blocks')
            return

        # otherwise create a new valid blocks json
        self.logger.log('INFO', 'Validating existing blocks')
        # prep block validation arguments
        all_blks = self.all_blocks
        jobs = [(_eval_blk, (b, px_thres), {}) for b in all_blks.items()]
        # parallel processing through all blocks
        results: list[dict] = utils.ParallelExecutor().run(jobs)
        self.valid_blks = {
            r['valid']: all_blks[r['valid']] for r in results if 'valid' in r
        }

        # log and save
        self.logger.log('INFO', f'Got {len(self.valid_blks)} valid blocks')
        self.logger.log('INFO', f'Valid blocks json saved to {_blks}')
        utils.write_json(_blks, self.valid_blks)

    # -----------------------------internal method-----------------------------
    def _prepare_block_windows(self):
        '''
        Intersect image and label windows and filter by expected shape.

        Keeps only coordinates present in both image and label windows
        and whose (width, height) match `windows.expected_shape`. The
        resulting set defines `self.all_blocks` with canonical block
        names mapped to their target `.npz` paths.
        '''

        # find shared coordinates between image and label read windows
        if self.has_label:
            self.shared_coords = set(self.windows.image_windows.keys()) \
                & set(self.windows.label_windows.keys())
        else:
            self.shared_coords = set(self.windows.image_windows.keys())
        self.logger.log('DEBUG', f'Loaded {len(self.shared_coords)} read windows')

        # remove windows or irregular shapes, e.g., edge windows
        for coord in self.shared_coords:
            # access image window
            iw = self.windows.image_windows[coord]
            if (iw.width, iw.height) != self.windows.expected_shape:
                self.shared_coords.remove(coord)
            if self.has_label:
                lw = self.windows.label_windows[coord]
                if (lw.width, lw.height) != self.windows.expected_shape:
                    self.shared_coords.remove(coord)
        self.logger.log('DEBUG', f'Windows with expected shape: {len(self.shared_coords)}')

        # all blocks come from the shared coordinates
        self.all_blocks = {
            self._xy_name(c): f'{self.config.blks_dpath}/{self._xy_name(c)}.npz'
            for c in self.shared_coords
        }
        # write an artifect: normal block list
        utils.write_json(self.config.all_blocks, self.all_blocks)

    def _validate_existing_blocks(self) -> None:
        '''
        Verify integrity of expected block '.npz' files.

        Runs parallel checks over all expected file paths. Any missing
        or corrupted files are reported; corrupted files that exist on
        disk are removed. Does not create new blocks.
        '''

        # check files from expected list of .npz files
        self.logger.log('INFO', 'Checking block .npz files')
        # parallel processing
        jobs = [(_check_npz, (f,), {}) for f in self.all_blocks.values()]
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

    def _create_missing_blocks(self) -> None:
        '''
        Create missing block '.npz' files.

        Determines which blocks are absent by comparing expected names
        to the current contents of 'output.blks_dpath'. For each missing
        block, reads the required rasters, constructs the data block,
        and writes it to disk. Finally writes `output.all_blocks` JSON.
        '''

        # determine block files that need to be created
        existing = set(os.listdir(self.config.blks_dpath)) # filenames
        coords_todo = set(
            self._name_xy(name) for name, path in self.all_blocks.items()
            if os.path.basename(path) not in existing # also filenames
        )
        if not coords_todo:
            self.logger.log('INFO', 'No data blocks to be created')
            return
        self.logger.log('INFO', f'{len(coords_todo)} data blocks to be created')

        # prep block creation jobs
        jobs = []
        meta_src: dataprep.BlockMeta = utils.load_json(self.config.config_fpath)
        for c in coords_todo:
            name = self._xy_name(c)
            co_contxt = _BlockCreationContext(
                name=name,
                meta=copy.deepcopy(meta_src),
                img_path=self.config.image_fpath,
                img_window=self.windows.image_windows[c],
                lbl_path=self.config.label_fpath,
                lbl_window=self.windows.label_windows[c] if self.has_label else None,
                npz_fpath=self.all_blocks[name]
            )
            jobs.append((_make_blk, (co_contxt,), {}))

        # parallel processing through all raster windows
        utils.ParallelExecutor().run(jobs)

    # -------------------------------properties-------------------------------
    @property
    def has_label(self) -> bool:
        '''If current pipeline is supplied with a label raster.'''

        return bool(self.config.label_fpath)

    # ------------------------------static method------------------------------
    @staticmethod
    def _xy_name(coords: tuple[int, int]) -> str:
        '''
        Convert (x, y) coordinates to a canonical block name string:
        `(12, 34)` -> `'col_000012_row_000034'`.
        '''

        x, y = coords
        return f'col_{x:06d}_row_{y:06d}'

    @staticmethod
    def _name_xy(name: str) -> tuple[int, int]:
        '''
        Convert a canonical block name back to coordinates:
        `'col_000012_row_000034'` -> `(12, 34)`.
        '''

        split = name.split('_')
        x_str, y_str = split[1], split[3]
        return int(x_str), int(y_str)

# ------------------------------private functions------------------------------
# outside of class for the use in parallel processing
def _check_npz(blk_fpath: str) -> dict[str, str]:
    '''Check if a .npz block file can be loaded properly.'''

    rb = dataprep.DataBlock()
    # pass if the npz file can be loaded properly
    try:
        rb.load(blk_fpath)
        return {'pass': blk_fpath}
    # flag absent/corrupted/damaged npz file
    except (FileNotFoundError, zipfile.error, zlib.error):
        return {'invalid': blk_fpath}

def _make_blk(contxt: _BlockCreationContext) -> None:
    '''Create a block from the input rasters for the given window.'''

    # parse args
    meta = contxt.meta
    img_path = contxt.img_path
    img_window = contxt.img_window
    lbl_path = contxt.lbl_path
    lbl_window = contxt.lbl_window
    npz_fpath = contxt.npz_fpath

    # meta i/o
    meta['block_name'] = contxt.name # assign name
    dem_band = meta['band_map']['dem']
    pad = meta['dem_pad']

    # read rasters at given window and create blocks
    with utils.open_rasters(img_path, lbl_path) as (img, lbl):
        # sanity check, image raster must be provided
        assert img is not None
        # read image array
        img_arr: numpy.ndarray = img.read(window=img_window, boundless=True)
        meta['image_nodata'] = img.nodata
        # get padded dem array from image
        padded_dem = _read_w_pad(img, img_window, dem_band, pad)
        # read label array if provided
        lbl_arr: numpy.ndarray | None = None
        if lbl is not None and lbl_window is not None:
            lbl_arr = lbl.read(window=lbl_window, boundless=True)
            meta['label_nodata'] = lbl.nodata

        # create and save
        blk = dataprep.DataBlock().build(img_arr, lbl_arr, padded_dem, meta)
        blk.save(npz_fpath)

def _read_w_pad(
    img: alias.RasterReader,
    window: alias.RasterWindow,
    dem_band: int,
    pad: int
) -> numpy.ndarray:
    '''Read the DEM band around 'window' with reflection padding.'''

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

def _eval_blk(
    block: tuple[str, str],
    valid_px_threshold: float,
) -> dict[str, str]:
    '''Evaluate a block's validity against a pixel-ratio threshold.'''

    # parse arguments
    name, fpath = block
    # get meta from block
    meta = dataprep.DataBlock().load(fpath).meta
    # valid pixel ratio threshold
    if meta['valid_pixel_ratio']['block'] < valid_px_threshold:
        return {'invalid': name}
    return {'valid': name}
