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
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BlockBuilderConfig:
    '''
    I/O paths and configuration parameters used during block construction.

    This configuration specifies input rasters, metadata sources, and the
    output catalog root. All values here are treated as static parameters
    for the block-building pipeline.
    '''
    image_fpath: str            # path to input image data (.tiff)
    label_fpath: str | None     # path to input label data (.tiff)
    config_fpath: str           # path to input metadata (.json)
    output_root: str            # path to output artifacts
    ignore_index: int           # global ignore label index
    dem_pad_px: int             # image DEM channel padding in pixels
    block_size: tuple[int, int] # block size in row, col

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
        image_windows: alias.RasterWindowDict,
        label_windows: alias.RasterWindowDict,
        config: BlockBuilderConfig,
        logger: utils.Logger,
    ):
        '''
        Initialize the pipeline.

        Args:
            image_windows: Image read windows.
            label_windows: Label read windows
            config: Block builder configuration.
            logger: Logger for progress and diagnostics.
        '''

        # intake arguments
        self.img_windows = image_windows
        self.lbl_windows = label_windows
        self.config = config
        self.logger = logger

        # declare class attributes
        self.common_coords: set[tuple[int, int]] = set()
        self.coords_todo: list[tuple[int, int]] = []

        # parse block meta dict (carried by each block)
        meta_src = utils.load_json(self.config.config_fpath)
        keys = meta_src.keys() & geo_core.BlockMeta.__annotations__
        meta = {k: meta_src[k] for k in keys}
        self.meta = typing.cast(geo_core.BlockMeta, meta) # typing compliance

        # make sure output dir for the blocks exist
        os.makedirs(self.blks_dir, exist_ok=True)

    @property
    def blks_dir(self) -> str:
        '''Directory to save `.npz` block files.'''
        return f'{self.config.output_root}/blocks'

    @property
    def catalog_path(self) -> str:
        '''File path where catalog JSON is to load/save.'''
        return f'{self.config.output_root}/catalog.json'

    @property
    def has_label(self) -> bool:
        '''If current pipeline is supplied with a label raster.'''
        label_fpath = self.config.label_fpath
        return bool(label_fpath) and os.path.exists(label_fpath)

    def build_single_block(
        self,
        save_dpath: str,
        *,
        valid_px_per: float,
        monitor_head: str,
        need_all_classes: bool
    ) -> str | None:
        '''
        Build, normalize, and persist a single valid data block.

        Iterates deterministically over grid coordinates to locate the
        first block satisfying the requested validity and label-coverage
        criteria. This path is intended for debugging and overfit testing
        only.

        Unlike the standard block-building pipeline, this method **does
        apply in-place image normalization** using per-block mean and
        standard deviation prior to saving. This behavior is intentional
        and scoped exclusively to single-block workflows.

        Args:
            save_dpath: Directory where the selected block is written.
            valid_px_per: Minimum fraction of valid pixels required for
                acceptance.
            monitor_head: Label head used to verify class presence.
            need_all_classes: If True, require all classes to be present
                in the monitored head.

        Returns:
            File path to the saved block if a valid block is found;
            otherwise None.
        '''

        # get a deterministic coordinate sequence to iterate
        coords = list(self.img_windows.keys()) # from image windows
        random.Random(42).shuffle(coords)

        # iterate through till a valid block if found
        c: tuple[int, int] = (0, 0)
        blk: geo_core.DataBlock | None = None
        for c in coords:
            print('Searching for a valid block...', end='\r', flush=True)
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
            return None

        # normalize the image array
        mean = numpy.mean(blk.data.image)
        std = numpy.std(blk.data.image)
        blk.data.image = (blk.data.image - mean) / (std or 1.0)

        # log and save block to location
        self.logger.log('INFO', f'Fetched a valid block at coord: {c}')
        self.logger.log('DEBUG', 'Criteria:')
        self.logger.log('DEBUG', f'Minimum valid pixel: {valid_px_per:.2f}')
        self.logger.log('DEBUG', f'Focused head: {monitor_head}')
        self.logger.log('DEBUG', f'Requires all classes: {need_all_classes}')
        os.makedirs(save_dpath, exist_ok=True) # safety
        fpath = f'{save_dpath}/{geo_utils.xy_name(c)}.npz'
        blk.save(fpath)
        # return the block fpath
        return fpath

    def build_blocks(self) -> list[tuple[int, int]]:
        '''
        Run the full construction sequence:

        1) Intersect and filter the read windows by expected shape.
        2) Validate existing `.npz` blocks; remove corrupted ones.
        3) Create `.npz` files only for missing or invalid blocks.

        Returns:
            coordinatess where blocks were created.
        '''

        # run building sequence
        self._prepare_block_windows()
        self._validate_existing_blocks()
        self._create_missing_blocks()
        # return coords where blocks were created
        return self.coords_todo

    # -----------------------------internal method-----------------------------
    def _prepare_block_windows(self) -> None:
        '''
        Intersect image and label windows and filter by expected shape.

        Defines the canonical set of coordinates used during validation,
        block creation, and catalog updates.
        '''

        # find shared coordinates between image and label read windows
        if self.has_label:
            self.common_coords = set(self.img_windows.keys()) \
                & set(self.lbl_windows.keys())
        else:
            self.common_coords = set(self.img_windows.keys())
        n = len(self.common_coords)
        self.logger.log('DEBUG', f'Loaded {n} raster windows')

        # remove windows or irregular shapes, e.g., edge windows
        for coord in self.common_coords:
            # access image window
            iw = self.img_windows[coord]
            if (iw.height, iw.width) != self.config.block_size:
                self.common_coords.remove(coord)
            if self.has_label:
                lw = self.lbl_windows[coord]
                if (lw.height, lw.width) != self.config.block_size:
                    self.common_coords.remove(coord)
        n = len(self.common_coords)
        self.logger.log('DEBUG', f'Number of windows with expected shape: {n}')

    def _validate_existing_blocks(self) -> None:
        '''
        Check integrity of expected block `.npz` files.

        Attempts to load each block and optionally verify its SHA-256
        hash. Corrupted or unreadable files are flagged, recorded for
        regeneration, and removed from disk. This step does not create
        new blocks.
        '''

        # block files to check from common coordinates
        blks_to_check: dict[tuple[int, int], str] = {}
        self.logger.log('INFO', 'Checking block .npz files')
        for c in self.common_coords:
            name = geo_utils.xy_name(c)
            blks_to_check[c] = f'{self.blks_dir}/{name}.npz'

        # create checking jobs
        jobs = [(_check_npz, (c, fp), {}) for c, fp in blks_to_check.items()]
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
        removed = 0
        for fpath in rm:
            try:
                os.remove(fpath)
                removed += 1
            except FileNotFoundError:
                continue

        # log checking results
        self.logger.log('INFO', f'Found {len(self.coords_todo)} invalid blocks')
        self.logger.log('INFO', f'Removed {removed} damaged files')

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
                'save_fpath': f'{self.blks_dir}/{geo_utils.xy_name(c)}.npz'
            }
            jobs.append((_build_a_blk, (meta, co_contxt,), save_args))

        # parallel processing through all raster windows
        utils.ParallelExecutor().run(jobs)

    def _get_context(self, coords: tuple[int, int]) -> _BlockCreationContext:
        '''Return a the immutable block-creation context.'''

        return _BlockCreationContext(
            name=geo_utils.xy_name(coords),
            ignore_index=self.config.ignore_index,
            dem_pad_px=self.config.dem_pad_px,
            img_path=self.config.image_fpath,
            img_window=self.img_windows[coords],
            lbl_path=self.config.label_fpath,
            lbl_window=self.lbl_windows[coords] if self.has_label else None
        )

# ------------------------------private functions------------------------------
# function outside of class for the use in parallel processing
def _check_npz(
    coord: tuple[int, int],
    fpath: str,
) -> dict[tuple[int, int], bool]:
    '''Verify whether a `.npz` block file can be successfully loaded.'''

    ok = False
    # pass if the npz file can be loaded properly
    try:
        geo_core.DataBlock.load(fpath)
        ok = True
    # flag absent/corrupted/damaged npz file
    except (FileNotFoundError, zipfile.error, zlib.error):
        ok = False
    return {coord: ok}

def _build_a_blk(
    meta: geo_core.BlockMeta,
    contxt: _BlockCreationContext,
    *,
    save: bool = False,
    save_fpath: str | None = None
) -> geo_core.DataBlock:
    '''Create a block from the input rasters for the given window.'''

    # meta i/o
    meta['block_name'] = contxt.name # assign name
    dem_band = meta['image_band_map']['dem']

    # read rasters at given window and create blocks
    with geo_utils.open_rasters(contxt.img_path, contxt.lbl_path) as (img, lbl):
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
    output_block = geo_core.DataBlock.build(img_arr, lbl_arr, padded_dem, meta)
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
    try:
        _window = alias.RasterWindow(nw_x, nw_y, se_x - nw_x, se_y - nw_y) # type: ignore
    except ValueError:
        print(nw_x, nw_y, se_x - nw_x, se_y - nw_y)
        raise

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
