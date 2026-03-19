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
Data block preparation and cataloguing pipeline.

This module consumes precomputed raster read windows and constructs a
disk-backed block cache. It no longer performs full block validation;
instead, it focuses on gap-filling and consistent catalog maintenance.

Primary responsibilities:
- Load and intersect image/label read windows produced upstream.
- Create `.npz` data blocks only where absent (gap filling only).
- Catalogue all blocks discovered on disk, appending new entries and
  recording required metadata.

Notes:
- Window tiling, alignment, and grid construction are handled upstream.
- Existing blocks are not revalidated or rebuilt.
- No class balancing, filtering, or content-based pruning is performed
  here; such logic belongs in downstream sampling or training policies.
'''

# standard imports
from __future__ import annotations
import copy
import dataclasses
import random
import os
import typing
import zipfile
import zlib
# third-party imports
import numpy
# local imports
import landseg.core.alias as alias
import landseg._ingest_dataset.canonical.align as align
import landseg._ingest_dataset.canonical.blocks as blocks
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BuilderConfig:
    '''
    I/O paths and configuration parameters used during block construction.

    This configuration specifies input rasters, metadata sources, and the
    output catalog root. All values here are treated as static parameters
    for the block-building pipeline.
    '''
    image_fpath: str            # path to input image data (.tiff)
    label_fpath: str | None     # path to input label data (.tiff)
    config_fpath: str           # path to input metadata (.json)
    catalog_root: str           # root directory to save related artifacts
    grid_id: str                # world grid that the rasters are mapped to
    ignore_index: int           # global ignore label index
    dem_pad_px: int             # image DEM channel padding in pixels

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass(frozen=True)
class _BlockCreationContext:
    '''Immutable, per-block inputs required to build a single block.'''
    name: str
    ignore_index: int
    dem_pad_px: int
    img_path: str
    img_window: alias.RasterWindow
    lbl_path: str | None
    lbl_window: alias.RasterWindow | None

# --------------------------------Public  Class--------------------------------
class BlockBuilder:
    '''
    Orchestrates the preparation and creation of block `.npz` files.

    Pipeline stages:
    - Load precomputed image/label read windows and filter them by shape.
    - Identify block coordinates and generate any missing `.npz` files.
    - Catalogue all discovered blocks, appending new entries as needed.

    Notes:
    - Window tiling and grid alignment are handled upstream.
    - Block creation is strictly gap-filling; existing files are not
    checked or overwritten.
    - No integrity verification, class balancing, or content filtering
    is performed here.
    '''

    def __init__(
        self,
        config: BuilderConfig,
        logger: utils.Logger,
    ):
        '''
        Initialize the pipeline.

        Args:
            windows: Image/label read windows and expected shape.
            config: Block builder configuration.
            logger: Logger for progress and diagnostics.
        '''

        # intake arguments
        self.config = config
        self.logger = logger

        # declare class attributes
        self.catalog: blocks.BlocksCatalog
        self.windows: align.DataWindows
        self.common_coords: set[tuple[int, int]] = set()
        self.coords_todo: list[tuple[int, int]] = []

        root = self.config.catalog_root
        # load catalog.json
        self.catalog = blocks.BlocksCatalog.from_json(f'{root}/catalog.json')
        self.logger.log('INFO', f'Read {len(self.catalog)} catalog entries')

        # load raster windows
        windows_pkl = f'{root}/windows/windows_{self.config.grid_id}.pkl'
        try:
            self.windows = utils.load_pickle(windows_pkl)
            self.logger.log('INFO', 'Block windows loaded')
        except FileExistsError:
            self.logger.log('ERROR', f'{windows_pkl} not found')
            raise

        # parse block meta dict (carried by each block)
        meta_src = utils.load_json(self.config.config_fpath)
        keys = meta_src.keys() & blocks.BlockMeta.__annotations__
        meta = {k: meta_src[k] for k in keys}
        self.meta = typing.cast(blocks.BlockMeta, meta) # typing compliance

        # make sure output dir for the blocks exist
        os.makedirs(self.blks_dir, exist_ok=True)

    @property
    def blks_dir(self) -> str:
        '''Directory to save `.npz` block files.'''
        return f'{self.config.catalog_root}/blocks'

    @property
    def has_label(self) -> bool:
        '''If current pipeline is supplied with a label raster.'''
        label_fpath = self.config.label_fpath
        return bool(label_fpath) and os.path.exists(label_fpath)

    def build_single_block(
        self,
        save_dpath: str,
        *,
        valid_px_per: float = 0.8,
        monitor_head: str = 'layer1',
        need_all_classes: bool = True
    ) -> None:
        '''
        Build and save a single data block with the given criteria.
        '''

        # get a deterministic coordinate sequence to iterate
        coords = list(self.windows.image.keys()) # from image windows
        random.Random(42).shuffle(coords)

        # iterate through till a valid block if found
        c: tuple[int, int] = (0, 0)
        blk: blocks.DataBlock | None = None
        for c in coords:
            print('Searching for a valid raster window...', end='\r', flush=True)
            try:
                co_contxt = self._get_context(c)
                blk = _build_a_blk(self.meta, co_contxt, save=False) # temp
                meta = blk.meta
                block_ratio_ok = meta["valid_ratios"]["block"] >= valid_px_per
                has_all_labels = all(meta["label_count"][monitor_head])
                if block_ratio_ok and (has_all_labels or not need_all_classes):
                    break
            except ValueError: # likely an empty window for the rasters
                continue
        # if no block found log a warning and return
        if not blk:
            self.logger.log('WARNING', 'No valid block for testing.')
            return
        # otherwise log and save block to location
        self.logger.log('INFO', f'Fetched a valid block at coord: {c}')
        self.logger.log('DEBUG', 'Criteria:')
        self.logger.log('DEBUG', f'Minimum valid pixel: {valid_px_per:.2f}')
        self.logger.log('DEBUG', f'Focused head: {monitor_head}')
        self.logger.log('DEBUG', f'Requires all classes: {need_all_classes}')
        fpath = f'{save_dpath}/{_xy_name(c)}.npz'
        blk.save(fpath)

    def build_blocks(self) -> None:
        '''
        Run the full construction sequence:

        1) Intersect and filter the read windows by expected shape.
        2) Validate existing `.npz` blocks; remove corrupted ones.
        3) Create `.npz` files only for missing or invalid blocks.
        4) Update `catalog.json` to include any newly created blocks.
        '''

        # run building sequence
        self._prepare_block_windows()
        self._validate_existing_blocks()
        self._create_missing_blocks()
        self._update_catalog()

    # -----------------------------internal method-----------------------------
    def _prepare_block_windows(self) -> None:
        '''
        Intersect image and label windows and filter by expected shape.

        Defines the canonical set of coordinates used during validation,
        block creation, and catalog updates.
        '''

        # find shared coordinates between image and label read windows
        if self.has_label:
            self.common_coords = set(self.windows.image.keys()) \
                & set(self.windows.label.keys())
        else:
            self.common_coords = set(self.windows.image.keys())
        n = len(self.common_coords)
        self.logger.log('DEBUG', f'Loaded {n} raster windows')

        # remove windows or irregular shapes, e.g., edge windows
        for coord in self.common_coords:
            # access image window
            iw = self.windows.image[coord]
            if (iw.width, iw.height) != self.windows.tile_shape:
                self.common_coords.remove(coord)
            if self.has_label:
                lw = self.windows.label[coord]
                if (lw.width, lw.height) != self.windows.tile_shape:
                    self.common_coords.remove(coord)
        n = len(self.common_coords)
        self.logger.log('DEBUG', f'Number of windows with expected shape: {n}')

    def _validate_existing_blocks(self, *, deep_scan: bool = False) -> None:
        '''
        Check integrity of expected block `.npz` files.

        Attempts to load each block and optionally verify its SHA-256
        hash. Corrupted or unreadable files are flagged, recorded for
        regeneration, and removed from disk. This step does not create
        new blocks.
        '''

        # block files to check from common coordinates
        blks_to_check: dict[tuple[int, int], str] = {}
        blks_in_catalog: dict[tuple[int, int], blocks.CatalogEntry | None] = {}
        self.logger.log('INFO', 'Checking block .npz files')
        for c in self.common_coords:
            name = _xy_name(c)
            blks_to_check[c] = f'{self.blks_dir}/{name}.npz'
            blks_in_catalog[c] = self.catalog.get(c)

        # create checking jobs
        jobs = []
        for coord, catalog_entry in blks_in_catalog.items():
            if catalog_entry:
                fpath = catalog_entry['file_path']
                kwargs = {
                    'sha_256': catalog_entry['sha_256'],
                }
                jobs.append((_check_npz, (coord, fpath, deep_scan), kwargs))
            else:
                fpath = blks_to_check[coord]
                jobs.append((_check_npz, (coord, fpath, False), {}))

        # parallel processing
        raw_results: list[dict[tuple[int, int], bool]]
        raw_results = utils.ParallelExecutor().run(jobs)
        results = {k: v for rr in raw_results for k, v in rr.items()}

        # parse results
        rm: list[str] = [] # damaged files to be removed
        for c, valid in results.items():
            if not valid:
                self.coords_todo.append(c)
                rm.append(blks_to_check[c])
        # remove corrupted/damaged files if present
        for fpath in rm:
            try:
                os.remove(fpath)
            except FileNotFoundError:
                continue

        # log checking results
        self.logger.log('INFO', f'Found {len(self.coords_todo)} invalid blocks')
        self.logger.log('INFO', f'Removed {len(rm)} damaged files')

    def _create_missing_blocks(self) -> None:
        '''
        Create `.npz` blocks at locations flagged as missing or invalid.

        Reads raster data for each required window, constructs a
        `DataBlock`, and writes it to disk. Does not overwrite existing
        valid blocks.
        '''

        if not self.coords_todo:
            self.logger.log('INFO', 'No data blocks to be created')
            return
        self.logger.log('INFO', f'{len(self.coords_todo)} blocks to be created')

        # prep block creation jobs
        jobs = []
        for c in self.coords_todo:
            meta = copy.deepcopy(self.meta)
            co_contxt = self._get_context(c)
            save_args = {
                'save': True,
                'save_fpath': f'{self.blks_dir}/{_xy_name(c)}.npz'
            }
            jobs.append((_build_a_blk, (meta, co_contxt,), save_args))

        # parallel processing through all raster windows
        utils.ParallelExecutor().run(jobs)

    def _update_catalog(self) -> None:
        '''
        Update/create `catalog.json` to all valid block files on disk.

        Any blocks found on disk but missing from the catalog are hashed,
        loaded for metadata, and appended as new catalog entries.
        '''

        # check existing block files on disk
        on_disk = [p for p in os.listdir(self.blks_dir) if p.endswith('npz')]
        assert on_disk, 'No .npz block files found' # quick sanity

        # check existing blocks in catalog
        in_catalog = [f'{p["block_name"]}.npz' for p in self.catalog.values()]

        # get the difference as ones that need to be catalogued
        to_catalog = set(on_disk) - set(in_catalog) # on disk but not in catalog
        if not to_catalog:
            self.logger.log('INFO', 'No update to catalog.json needed')
            return

        self.logger.log('INFO', 'Creating/updating catalog.json...')
        # get hash values from input rasters
        img_hash = utils.hash_artifacts(self.config.image_fpath, False)
        if self.config.label_fpath:
            lbl_hash = utils.hash_artifacts(self.config.label_fpath, False)
        else:
            lbl_hash = None

        # add to current catalog dict
        for fname in to_catalog:
            fp = f'{self.blks_dir}/{fname}'
            meta = blocks.DataBlock.load(fp).meta
            col, row = _name_xy(meta['block_name'])
            self.catalog[(col, row)] = {
                'block_name': meta['block_name'],
                'file_path': fp,
                'loc_x_px': col,
                'loc_y_px': row,
                'valid_px': meta['valid_ratios']['layer1'],
                'class_count': meta['label_count']['layer1'],
                'schema_version': '1.0.0',
                'creation_time': utils.get_file_ctime(fp, '%Y-%m-%dT%H:%M:%S'),
                'sha_256': utils.hash_artifacts(fp, False),
                'aligned_grid': 'grid_row_256_128_col_256_128',
                'source_image': self.config.image_fpath,
                'source_image_sha_256': img_hash,
                'source_label': self.config.label_fpath,
                'source_label_sha_256': lbl_hash,
            }
        self.catalog.save_json(f'{self.config.catalog_root}/catalog.json')

    def _get_context(self, coords: tuple[int, int]) -> _BlockCreationContext:
        '''Return a the immutable block-creation context.'''

        return _BlockCreationContext(
            name=_xy_name(coords),
            ignore_index=self.config.ignore_index,
            dem_pad_px=self.config.dem_pad_px,
            img_path=self.config.image_fpath,
            img_window=self.windows.image[coords],
            lbl_path=self.config.label_fpath,
            lbl_window=self.windows.label[coords] if self.has_label else None
        )

# ------------------------------private functions------------------------------
# outside of class for the use in parallel processing
def _check_npz(
    coord: tuple[int, int],
    fpath: str,
    deep_check: bool = False,
    *,
    sha_256: str | None = None
) -> dict[tuple[int, int], bool]:
    '''
    Verify whether a `.npz` block file can be successfully loaded.

    Optionally performs a deep check by comparing the file's SHA-256 hash.
    Returns a mapping `{coord: is_valid}` indicating load/validation
    status.
    '''

    ok = False
    # pass if the npz file can be loaded properly
    try:
        blocks.DataBlock.load(fpath)
        # branch out if deep check
        if deep_check:
            blk_sha_256 = utils.hash_artifacts(fpath, write_to_record=False)
            if blk_sha_256 == sha_256:
                ok = True
        else:
            pass
        ok = True
    # flag absent/corrupted/damaged npz file
    except (FileNotFoundError, zipfile.error, zlib.error):
        ok = False

    return {coord: ok}

def _build_a_blk(
    meta: blocks.BlockMeta,
    contxt: _BlockCreationContext,
    *,
    save: bool = False,
    save_fpath: str | None = None
) -> blocks.DataBlock:
    '''Create a block from the input rasters for the given window.'''

    # meta i/o
    meta['block_name'] = contxt.name # assign name
    dem_band = meta['image_band_map']['dem']

    # read rasters at given window and create blocks
    with utils.open_rasters(contxt.img_path, contxt.lbl_path) as (img, lbl):
        # sanity check, image raster must be provided
        assert img is not None
        # read image array
        img_window = contxt.img_window
        img_arr: numpy.ndarray = img.read(window=img_window, boundless=True)
        meta['image_nodata'] = img.nodata
        # get padded dem array from image
        padded_dem = _read_w_pad(img, img_window, dem_band, contxt.dem_pad_px)
        # read label array if provided
        lbl_window = contxt.lbl_window
        lbl_arr: numpy.ndarray | None = None
        if lbl is not None and lbl_window is not None:
            lbl_arr = lbl.read(window=lbl_window, boundless=True)
            meta['label_nodata'] = lbl.nodata

    # create and return DataBlock instance
    output_block = blocks.DataBlock.build(img_arr, lbl_arr, padded_dem, meta)
    # by default save to provided target path
    if save:
        assert save_fpath
        output_block.save(save_fpath)
    return output_block

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

# helpers
def _xy_name(coords: tuple[int, int]) -> str:
    '''
    Convert (x, y) coordinates to a canonical block name string:
    `(12, 34)` -> `'col_000012_row_000034'`.
    '''

    x, y = coords
    return f'col_{x:06d}_row_{y:06d}'

def _name_xy(name: str) -> tuple[int, int]:
    '''
    Convert a canonical block name back to coordinates:
    `'col_000012_row_000034'` -> `(12, 34)`.
    '''

    split = name.split('_')
    x_str, y_str = split[1], split[3]
    return int(x_str), int(y_str)
