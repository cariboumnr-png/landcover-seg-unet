'''
Raster block layout utilities.

Provides `BlockLayout`, which ingests image and optional label rasters
and produces a tiling layout of square raster windows. The resulting
layout and metadata are consumed by downstream block cache pipelines.
'''

# standard imports
import dataclasses
import math
import typing
# third-party imports
import numpy
import rasterio
import rasterio.coords
import rasterio.transform
# local imports
import _types
import utils

# --------------------------------Public  Class--------------------------------
class BlockLayout:
    '''
    Create a tiling layout from image and optional label rasters.

    This class validates raster compatibility (CRS, pixel size, extent),
    computes a shared spatial extent, and divides it into square blocks
    with optional overlap.

    The primary entry point is `ingest()`, which populates:
      - `self.blks`: mapping of block names to raster windows
      - `self.meta`: layout metadata describing geometry and grid
    '''

    def __init__(
        self,
        blk_size: int,
        overlap: int,
        logger: utils.Logger
    ):
        '''
        Creates processing blocks from the input rasters.

        Args:
            block_size: The pixel size of the squared blocks.
            overlap: The overlap in pixel between blocks
            logger: Class-level logger for layout diagnostic.
        Raises:
            ValueError: if `overlap >= blk_size`.
        '''

        # assign attributes
        self.blk_size = blk_size
        self.overlap = overlap
        self.logger = logger

        # sanity
        if not blk_size > overlap:
            raise ValueError('Overlap must be smaller than block size.')

        # init attributes
        self.img: _types.RasterReader | None = None
        self.lbl: _types.RasterReader | None = None
        self.blks: dict[str, _types.RasterWindow] = {}
        self.meta: dict[str, typing.Any] = {}

    # -----------------------------public methods-----------------------------
    def ingest(
        self,
        image_fpath: str,
        label_fpath: str | None
    ) -> None:
        '''
        Ingest raster inputs and generate a block layout.

        This method:
        - validates CRS and pixel size
        - computes shared extent and transform
        - tiles the extent into square raster windows

        Raster reader handles are released before returning to allow
        pickling of the class instance.

        Args:
            image_fpath: Path to the image raster.
            label_fpath: Optional label raster path. If provided must be
                co-registered with the image raster.
        '''

        with utils.open_rasters(image_fpath, label_fpath) as (img, lbl):
            # assign attrs
            self.img = img
            self.lbl = lbl
            if self.lbl is None:
                self.meta['has_label'] = False
            else:
                self.meta['has_label'] = True
            # check if both rasters have the same projection system
            self._check_raster_proj()
            # check if both rasters have the same squared pixels
            self._check_raster_pixels()
            # get the overlapping extent from the input rasters
            self._get_overlap_extent()
            # get the new Affine transform needed for writing output raster
            self._get_new_transform()
            # tile the raster into blocks
            self._extent_to_blocks()

        # clear rasterio DatasetReader objects so the class can be pickled
        self.img = None
        self.lbl = None

    def reset(self) -> None:
        '''Clear all stored blocks and layout metadata.'''

        self.img = None
        self.lbl = None
        self.blks.clear()
        self.meta.clear()

    # ----------------------------internal methods----------------------------
    def _check_raster_proj(self) -> None:
        '''
        Check if the input rasters have the same projection.

        Raises:
            ValueError: If the input rasters do not have the same
                projection system.
        ----------------------------------------------------------------
        Populates `meta['projection']`.
        '''

        # if both image and label provided
        if self.img is not None and self.lbl is not None:
            self.logger.log('DEBUG', ' | Both image & label rasters provided')
            # get projection names, raster.crs might return differently
            try:
                crs_1 = self.img.crs.to_string().split('"')[1]
                crs_2 = self.lbl.crs.to_string().split('"')[1]
            except IndexError:
                crs_1 = self.img.crs
                crs_2 = self.lbl.crs

            # check if the projection systems are the same
            if crs_1 != crs_2:
                m = f' | Projections do not match: \n1: {crs_1} != 2: {crs_2}'
                self.logger.log('ERROR', m)
                raise ValueError('The rasters must have the same projection')

            # if both projections are the same
            self.meta['projection'] = crs_1
            self.logger.log('DEBUG', f' | Matching projections: {crs_1}')
        # or only image provided
        elif self.img is not None and self.lbl is None:
            try:
                crs_1 = self.img.crs.to_string().split('"')[1]
            except IndexError:
                crs_1 = self.img.crs
            self.meta['projection'] = crs_1
            self.logger.log('INFO', f'CRS from image raster: {crs_1}')
        # otherwise throw (once at this method)
        else:
            raise ValueError('Unrecognized raster types')

    def _check_raster_pixels(self) -> None:
        '''
        Check if the input rasters have the same squared pixels.

        This function uses affine transform get pixel resolution in
        both x and y directions and checks:

        * If both rasters have the same pixel size in both directions.
        * If the pixels are squared (pixel size in `x` == that in `y`).

        Raises:
            ValueError: If the pixel sizes are different or the pixels
                are not squared.
        ----------------------------------------------------------------
        Populates `meta['pixel_size']`.
        '''

        # if both image and label provided
        if self.img is not None and self.lbl is not None:
            # get the transform (Affine matrix) from the metadata
            transform_1 = self.img.transform
            transform_2 = self.lbl.transform

            # transform[0]: pixel size in the x direction (horizontal).
            # transform[4]: pixel size in the y direction (vertical).
            # Note: y is typically negative as the axis increases downward.
            x1, y1 = transform_1[0], -transform_1[4]
            x2, y2 = transform_2[0], -transform_2[4]

            # check if the pixels are squared (x == y)
            if x1 != y1 or x2 != y2:
                m = f' | Input rasters do not have the same squared pixels: '\
                    f'Raster1: ({x1}, {y1}), Raster2: ({x2}, {y2})'
                self.logger.log('ERROR', m)
                raise ValueError('Input rasters must have squared pixels')

            # check if the pixel sizes match
            if (x1, y1) != (x2, y2):
                m = f' | Input rasters have different pixel sizes: '\
                    f'Raster1: ({x1}, {y1}), Raster2: ({x2}, {y2})'
                self.logger.log('ERROR', m)
                raise ValueError('Input rasters must have the same pixel size')

            # assign value and log out
            self.meta['pixel_size'] = (x1, x1)
            self.logger.log('DEBUG', f' | Matching pixel sizes: {x1}*{x1}')

        # or only image provided
        elif self.img is not None and self.lbl is None:
            transform_1 = self.img.transform
            x1, y1 = transform_1[0], -transform_1[4]
            if x1 != y1:
                m = f' | Image raster do not have squared pixels: ({x1}*{y1})'
                self.logger.log('ERROR', m)
                raise ValueError('Image raster must have squared pixels')

            # assign value and log out
            self.meta['pixel_size'] = (x1, x1)
            self.logger.log('DEBUG', f' | Image raster pixel size: {x1}*{x1}')

    def _get_overlap_extent(self) -> None:
        '''
        Get the overlapping extent of the input rasters.

        The extent is defined by:
        * max of the left bounds.
        * max of the bottom bounds.
        * min of the right bounds.
        * min of the top bounds.

        Raises:
            ValueError: If input rasters have no overlapping extents.
        ----------------------------------------------------------------
        Populates `meta['raster1_bbox']`,` meta['raster2_bbox']`,
        `meta['same_bbox']`, and `meta['shared_bbox']`.
        '''

        # if both image and label provided
        if self.img is not None and self.lbl is not None:

            # get the bounding boxes
            b1 = self.img.bounds
            b2 = self.lbl.bounds

            # bounds(0-3) correspond to [left, bottom, right, top]
            lft = max(b1[0], b2[0]) # max of the left bounds
            btm = max(b1[1], b2[1]) # max of the bottom bounds
            rgt = min(b1[2], b2[2]) # min of the right bounds
            top = min(b1[3], b2[3]) # min of the top bounds

            # if the two do not overlop
            if lft >= rgt or btm >= top:
                self.logger.log('ERROR', ' | Input rasters have no overlaps')
                raise ValueError('Input rasters must have overlapping extents')

            # get the overlapping extent if no error
            bb = rasterio.coords.BoundingBox(lft, btm, rgt, top)

            # add bounding boxes to self
            self.meta['raster1_bbox'] = b1
            self.meta['raster2_bbox'] = b2
            self.meta['same_bbox'] = b1 == b2
            self.meta['shared_bbox'] = bb

        # or only image provided
        elif self.img is not None and self.lbl is None:

            self.meta['raster1_bbox'] = self.img.bounds
            self.meta['raster2_bbox'] = None
            self.meta['same_bbox'] = None
            self.meta['shared_bbox'] = self.img.bounds

    def _get_new_transform(self) -> None:
        '''
        Create Affine transform from the overlapping extent.\n
        ----------------------------------------------------------------
        Popluates `meta['transform']`.
        '''

        # get the boundaries and resolution
        l, _, _, t = self.meta['shared_bbox']
        p = self.meta['pixel_size'][0]
        # create the new transform
        self.meta['transform'] = rasterio.transform.from_origin(l, t, p, p)
        # log out
        self.logger.log('DEBUG', ' | Generating new transform ... OK')

    def _extent_to_blocks(self) -> None:
        '''
        Divide the overlapping extent to raster blocks.

        This function takes in a `rasterio.coords.BoundingBox` generated
        by `_get_overlap_extent()` and  calculates the number of pixels
        along the x and y axes based on the provided raster resolution
        and the extent. It divides the extent into square blocks of the
        given `block_size`, from top-left to bottom-right. If the extent
        cannot be evenly divided, right and bottom edge blocks will be
        created. For example:

        * A 100x100 pixel extent divided into 20x20 pixel blocks
        results in exactly 25 blocks.
        * A 110x105 pixel extent divided into 20x20 pixel blocks
        resutls in 25 regular 20x20 blocks, 5 right edge blocks of
        20x5 pixels, 5 bottom edge blocks of 10x20 pixels, and 1
        bottom-right corner block of 10x5 pixels.

        The blocks are indexed by the row and col numbers from top-left
        to bottom-right, e.g., the top-left block will be indexed as
        `[row_0, col_0]`.

        Intended to work on projected rasters with meter unit, such as
        an UTM projected raster.\n
        ----------------------------------------------------------------
        Populates `meta['H']`, `meta['W']`, `meta['regular_rows']`,
        `meta['regular_cols']`, `meta['r_edge_px']`, `meta['b_edge_px']`,
        `meta['all_rows']`, `meta['all_cols']`,` meta['total_blks']`,
        and `meta['regular_blks']`
        '''

        # get extent dimensions (pixel count) - round to nearest int
        l, b, r, t = self.meta['shared_bbox']
        p = self.meta['pixel_size'][0]
        h = math.floor((t - b) / p) # h = T - B / pixel size
        w = math.floor((r - l) / p) # w = R - L / pixel size

        # iterate through the blocks by row then col
        step = self.blk_size - self.overlap
        for i in range(0, h, step):
            for j in range(0, w, step):
                # create a block index from its position
                idx = _BlockName(j, i).name
                # dynamically adjust window size to stay within bounds
                bh = min(self.blk_size, h - i) # at the last row
                bw = min(self.blk_size, w - j) # at the last col
                # set up the window and update the result dict
                read_window = _types.RasterWindow(j, i, bw, bh) # type: ignore
                if self.__check_valid(read_window):
                    self.blks[idx] =  read_window

        # extent dimension
        self.meta['H'] = h
        self.meta['W'] = w

        # grid(row, col) of regular blocks
        self.meta['reg_rows'] = h // step
        self.meta['reg_cols'] = w // step

        # overhang pixels at bottom and right edges
        self.meta['b_edge_px'] = h % step
        self.meta['r_edge_px'] = w % step

        # grid (row, col) of all blocks
        self.meta['all_rows'] = self.meta['reg_rows'] + int(h % step > 0)
        self.meta['all_cols'] = self.meta['reg_cols'] + int(w % step > 0)

        # block count:
        self.meta['total_blks'] = self.meta['all_rows'] * self.meta['all_cols']
        self.meta['reg_blks'] = self.meta['reg_rows'] * self.meta['reg_cols']
        self.meta['valid_blks'] = len(self.blks)

    def __check_valid(self, window: _types.RasterWindow) -> bool:
        '''Crude check if a `Window` yields valid data from rasters.'''

        # read the rasters at the current block
        assert self.img is not None
        img_data: numpy.ndarray = self.img.read(1, window=window)
        lbl_data: numpy.ndarray | None = None
        if self.lbl is not None:
            lbl_data = self.lbl.read(1, window=window)

        # check validity
        # if the block is sqaure
        if window.height != window.width:
            return False
        # if image and label raster both present
        if numpy.any(img_data != self.img.nodata) and \
            (self.lbl is not None and numpy.any(lbl_data != self.lbl.nodata)):
            return True
        # if only image present
        if numpy.any(img_data != self.img.nodata) and self.lbl is None:
            return True
        # otherwise
        return False

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _BlockName:
    '''Simple class to define block name from column and row number.'''
    col: int
    row: int
    colrow: tuple[int, int] = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.colrow = (self.col, self.row)
        self.name = f'col_{self.col}_row_{self.row}'
