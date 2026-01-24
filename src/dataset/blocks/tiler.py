'''
Public Class:
    RasterBlocks(): Generates and stores blocks from two input rasters.
'''
from __future__ import annotations
# standard imports
import dataclasses
import re
import typing
# third-party imports
import numpy
import rasterio
import rasterio.coords
import rasterio.io
import rasterio.transform
import rasterio.windows
# local imports
import utils

# typing aliases
DatasetReader: typing.TypeAlias = rasterio.io.DatasetReader
Window: typing.TypeAlias = rasterio.windows.Window

class RasterPairedBlocks(dict):
    '''
    A class to ingest two input rasters and tile them into blocks.
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
            logger: Class-level logger.
        '''

        # assign attributes
        self.blk_size = blk_size
        self.overlap = overlap
        self.step = blk_size - overlap
        self.logger = logger

        # sanity
        assert blk_size > overlap, 'Overlap must be smaller than block size.'

        # init attributes
        self.ras_1: DatasetReader | None = None
        self.ras_2: DatasetReader | None = None

    # public methods
    def ingest(self, rasters: tuple[str, str]) -> None:
        '''
        Stage the class instance for further raster calculations.

        rasters: A tuple of two file paths to the input rasters.
        '''

        # open the input rasters
        with rasterio.open(rasters[0]) as ras_1, \
             rasterio.open(rasters[1]) as ras_2:
            # ras_1 and ras_2 passed as class attr for followed usage
            self.ras_1 = ras_1
            self.ras_2 = ras_2
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
            # check and mark for blocks where rasters are empty
            self.__check_empty_blocks()

        # clears raster reader attributes so the class instance can be copied
        self.ras_1 = None
        self.ras_2 = None

    @property
    def metadata(self) -> dict[str, typing.Any]:
        '''Raster pair metadata.'''

        return {
            'projection': self['proj'],
            'pixel_size': (self['res'], self['res']),
            'bounding_boxes': {
                'raster1_bbox': self['raster1_bbox'],
                'raster2_bbox': self['raster2_bbox'],
                'have_same_bbox': self['same_bbox'],
                'shared_bbox': self['shared_bbox']
            },
            'transform': self['transform'],
            'extent': (self['H'], self['W']),
            'block_counts': {
                'total_blocks': self['total_blks'],
                'regular_blocks': self['regular_blocks']
            },
            'block_grid': {
                'regular_rows': self['regular_rows'],
                'regular_cols': self['regular_cols'],
                'all_rows': self['all_rows'],
                'all_cols': self['all_cols']
            },
            'edge_pixels':{
                'right_edge': self['r_edge_px'],
                'bottom_edge': self['b_edge_px']
            }
        }

    def _check_raster_proj(self) -> None:
        '''
        Check if the input rasters have the same projection.

        Raises:
            ValueError: If the input rasters do not have the same
                projection system.
        ----------------------------------------------------------------
        Populate key `'proj'`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # get projection names, raster.crs might return differently
        try:
            crs_1 = self.ras_1.crs.to_string().split('"')[1]
            crs_2 = self.ras_2.crs.to_string().split('"')[1]
        except IndexError:
            crs_1 = self.ras_1.crs
            crs_2 = self.ras_2.crs

        # check if the projection systems are the same
        if crs_1 != crs_2:
            m = f' | Projections do not match: \n1: {crs_1} != 2: {crs_2}'
            self.logger.log('ERROR', m)
            raise ValueError(' | Both rasters must have the same projection')

        # if both projections are the same
        self['proj'] = crs_1
        self.logger.log('DEBUG', ' | Checking if the same projection ... OK')

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
        Populate key `'res'`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # get the transform (Affine matrix) from the metadata
        transform_1 = self.ras_1.transform
        transform_2 = self.ras_2.transform

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
            raise ValueError(' | Input rasters must have squared pixels')

        # check if the pixel sizes match
        if (x1, y1) != (x2, y2):
            m = f' | Input rasters have different pixel sizes: '\
                f'Raster1: ({x1}, {y1}), Raster2: ({x2}, {y2})'
            self.logger.log('ERROR', m)
            raise ValueError(' | Input rasters must have the same pixel sizes')

        # assign value and log out
        self['res'] = x1
        self.logger.log('DEBUG', ' | Checking if the same pixels ... OK')

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
        Populate keys: `'raster1_bbox'`, `'raster2_bbox'`, `'same_bbox'`,
        and `'shared_bbox'`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # get the bounding boxes
        b1 = self.ras_1.bounds
        b2 = self.ras_2.bounds

        # bounds(0-3) correspond to [left, bottom, right, top] side of the box
        lft = max(b1[0], b2[0]) # max of the left bounds
        btm = max(b1[1], b2[1]) # max of the bottom bounds
        rgt = min(b1[2], b2[2]) # min of the right bounds
        top = min(b1[3], b2[3]) # min of the top bounds

        # if the two do not overlop
        if lft >= rgt or btm >= top:
            self.logger.log('ERROR', ' | Input rasters have no overlaps')
            raise ValueError(' | Input rasters must have overlapping extents')

        # get the overlapping extent if no error
        bb = rasterio.coords.BoundingBox(lft, btm, rgt, top)

        # add bounding boxes to self
        self['raster1_bbox'] = b1
        self['raster2_bbox'] = b2
        self['same_bbox'] = b1 == b2
        self['shared_bbox'] = bb

    def _get_new_transform(self) -> None:
        '''
        Create Affine transform from the overlapping extent.\n
        ----------------------------------------------------------------
        Popluate key: `transform`.
        '''

        # get the boundaries and resolution
        left, _, _, top = self['shared_bbox']
        res = self['res']
        # create the new transform
        self['transform'] = rasterio.transform.from_origin(left, top, res, res)
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
        Populate keys: `'H'`, `'W'`, `'regular_rows'`, `'regular_cols'`,
        `'r_edge_px'`, `'b_edge_px'`, `'all_rows'`, `'all_cols'`,
        `'total_blks'`, and `'regular_blks'`.
        '''

        # get extent dimensions (pixel count) - round to nearest int
        # h = T - B
        extent_h = round((self['extent'][3] - self['extent'][1]) / self['res'])
        # w = R - L
        extent_w = round((self['extent'][2] - self['extent'][0]) / self['res'])

        # iterate through the blocks by row then col
        for i in range(0, extent_h, self.step):
            for j in range(0, extent_w, self.step):
                # create a block index from its position
                idx = BlockName(j, i).name
                # dynamically adjust window size to stay within bounds
                blk_h = min(self.blk_size, extent_h - i) # at the last row
                blk_w = min(self.blk_size, extent_w - j) # at the last col
                # set up the window and update the result dict
                self['all_blocks'][idx] = Window(j, i, blk_w, blk_h) # type: ignore

        # add meta to self
        # extent dimension
        self['H'] = extent_h
        self['W'] = extent_w

        # grid(row, col) of regular blocks
        self['regular_rows'] = self['H'] // self.step
        self['regular_cols'] = self['W'] // self.step

        # overhang(R and B edges) pixels
        self['r_edge_px'] = self['H'] % self.step
        self['b_edge_px'] = self['W'] % self.step

        # grid(row, col) of all blocks
        self['all_rows'] = self['regular_rows'] + 1 \
            if self['r_edge_px'] > 0 else self['regular_rows']
        self['all_cols'] = self['regular_cols'] + 1 \
            if self['b_edge_px'] > 0 else self['regular_cols']

        # block count:
        self['total_blks'] = self['all_rows'] * self['all_cols']
        self['regular_blks'] = self['regular_rows'] * self['regular_cols']

    def __check_empty_blocks(self) -> None:
        '''
        Check for blocks where the rasters are of `nodata`.

        This function go through the blocks generated by the function
        `_extent_to_blocks()` and identify:

        * blocks where both rasters are valid.
        * blocks where only raster_1 is empty.
        * blocks where only raster_2 is empty.
        * blocks where both rasters are empty.

        While calculation is needed for the first category, the rest can
        be simply assigned a value of choice, saving overall computing.

        ----------------------------------------------------------------
        Populate keys: `'blocks_both_valid'`, `'blocks_ras1_empty'`,
        `'blocks_ras2_empty'`, `'blocks_both_empty'`, and
        `'blocks_some_empty'`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # create dictionaries
        dict_1 = {} # blocks where both rasters are valid
        dict_2 = {} # blocks where only raster_1 is empty
        dict_3 = {} # blocks where only raster_2 is empty
        dict_4 = {} # blocks where both rasters are empty

        # iterate through blocks
        for idx, block in self['all_blocks'].items():

            # read the rasters at the current block
            raster_data_1 = self.ras_1.read(1, window=block)
            raster_data_2 = self.ras_2.read(1, window=block)

            # both are not empty
            if numpy.any(raster_data_1 != self.ras_1.nodata) and \
                numpy.any(raster_data_2 != self.ras_2.nodata):
                dict_1[idx] = block

            # only raster_1 is empty
            if numpy.all(raster_data_1 == self.ras_1.nodata) and \
                numpy.any(raster_data_2 != self.ras_2.nodata):
                dict_2[idx] = block

            # only raster_2 is empty
            if numpy.any(raster_data_1 != self.ras_1.nodata) and \
                numpy.all(raster_data_2 == self.ras_2.nodata):
                dict_3[idx] = block

            # both are empty
            if numpy.all(raster_data_1 == self.ras_1.nodata) and \
                numpy.all(raster_data_2 == self.ras_2.nodata):
                dict_4[idx] = block

        # assign values
        self['blocks_both_valid'] = dict_1
        self['blocks_ras1_empty'] = dict_2
        self['blocks_ras2_empty'] = dict_3
        self['blocks_both_empty'] = dict_4
        self['blocks_some_empty'] = [dict_2, dict_3, dict_4] # a list of dicts

@dataclasses.dataclass
class BlockName:
    '''Simple class to define block name from column and row number.'''
    col: int
    row: int
    colrow: tuple[int, int] = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.colrow = (self.col, self.row)
        self.name = f'col_{self.col}_row_{self.row}'


def parse_block_name(input_str: str) -> BlockName:
    '''Retrieve block naming info from a given string.'''

    # find pattern from string
    pattern = r'col_(\d+)_row_(\d+)'
    matched = re.search(pattern, input_str)
    # there should be just one match
    if not matched:
        raise ValueError(f'Block naming pattern {pattern} not found')
    # get col and row
    col = int(matched.group(1))
    row = int(matched.group(2))
    # return
    return BlockName(col, row)
