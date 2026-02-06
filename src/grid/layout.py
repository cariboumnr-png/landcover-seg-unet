# pylint: disable=c-extension-no-member
'''
World-grid tiling utilities.

This module is responsible only for defining a stable, versioned world
grid and producing a fixed set of pixel-aligned tile windows anchored in
a projected CRS. The grid itself is immutable and independent of any
specific raster dataset.

This module does NOT perform raster reprojection, resampling, or pixel
alignment. Users must ensure that input rasters are already aligned to
the grid in terms of CRS, pixel size, and pixel origin before attempting
to snap or read data using the grid windows.

In particular, this module assumes:
- Input rasters share the same projected CRS as the grid.
- Input rasters have identical pixel size.
- Raster pixel origins are aligned to the grid origin on integer pixel
  boundaries.

If raster pixels are not aligned to the grid (e.g., same pixel size but
origin offsets not divisible by pixel size), alignment must be resolved
upstream using external reprojection or snapping tools (e.g., GDAL,
QGIS, ArcGIS).

Recommended workflow:
- Create a blank or reference raster covering the study area with a
  stable CRS, pixel size, and origin.
- Define the world grid from this reference raster.
- Snap all training, inference, and ancillary rasters to the reference
  raster using external tools.
- Use this module only for deterministic grid indexing and window
  generation.
'''

# standard imports
from __future__ import annotations
import collections.abc
import dataclasses
import math
# third-party imports
import rasterio
import rasterio.crs
import rasterio.io
import rasterio.windows

# --------------------------------Public  Class--------------------------------
class GridLayout(collections.abc.Mapping[tuple[int, int], rasterio.windows.Window]):
    '''
    Raster-agnostic grid layout as a dictionary of rasterio windows.
    '''

    def __init__(
        self,
        mode: str,
        spec: GridSpec
    ):
        '''
        Initiate the dict instance by the input `GridSpec` dataclass.
        '''

        # ingest spec and init attributes
        self.mode = mode
        self.spec = spec
        self._extent: tuple[int, int] = (0, 0) # (rows, cols)
        self._data: dict[tuple[int, int], rasterio.windows.Window] = {}
        self._offset_px: tuple[int, int] = (0, 0)  # (dc_px, dr_px)
        self._generate()

    def __getitem__(self, idx: tuple[int, int]) -> rasterio.windows.Window:
        # get base window
        base = self._data[idx]
        # get right/down offset (in pixels)
        dx, dy = self._offset_px
        # get components for raster window
        xoff, yoff = base.col_off - dx, base.row_off - dy
        width, height = base.width, base.height
        return rasterio.windows.Window(xoff, yoff, width, height) # type: ignore

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def align_to_raster(self, src: rasterio.io.DatasetReader):
        '''
        Align the grid to a raster by computing an integer pixel offset.

        The raster must share the same CRS and pixel size as the grid.
        The offset is computed from the difference between grid origin
        and raster upper-left corner, expressed in pixels.

        After alignment, grid tile indices map consistently to raster
        pixel windows.

        Args:
            src: Input raster reader handler. Must align with the grid's
            CRS and pixel size.
        '''

        # check target raster CRS
        grid_crs = rasterio.crs.CRS.from_user_input(self.crs)
        inpt_crs = rasterio.crs.CRS.from_user_input(src.crs)
        assert grid_crs == inpt_crs
        # get raster origin in CRS units
        rx, ry = src.transform.c, src.transform.f
        # get raster pixel size and check alignment with the grid
        res_x, res_y = src.transform.a, abs(src.transform.e)
        assert self.spec.pixel_size[0] == res_x
        assert self.spec.pixel_size[1] == res_y
        # get world grid origin in CRS units
        gx, gy = self.spec.origin
        # calculate origin offset in pixel
        dc = math.floor((rx - gx) / res_x)      # + right
        dr = math.floor((ry - gy) / res_y)      # + down
        self._offset_px = (dc, dr)

    @property
    def crs(self) -> str:
        '''CRS of the layout.'''
        return self.spec.crs

    @property
    def origin(self) -> tuple[float, float]:
        '''Grid origin in CRS units (Easting, Northing).'''
        return self.spec.origin

    @property
    def tile_size(self) -> tuple[int, int]:
        '''Grid tile size in pixels.'''
        return self.spec.tile_size

    @property
    def tile_overlap(self) -> tuple[int, int]:
        '''Grid tile overlap in pixels.'''
        return self.spec.tile_overlap

    @property
    def h(self) -> int:
        '''Grid extent height in pixels.'''
        return self._extent[0]

    @property
    def w(self) -> int:
        '''Grid extent width in pixels.'''
        return self._extent[1]

    # -----------------------------private  method-----------------------------
    def _generate(self) -> None:
        '''
        Derive spatial extent from inputs and divide it into a grid.

        Intended to work with a projected CRS with meter unit.
        '''

        spec = self.spec
        # get extent dimensions (in crs units)
        if self.mode == 'aoi':
            assert spec.grid_extent is not None
            row_px = math.floor(spec.grid_extent[0] / spec.pixel_size[1])
            col_px = math.floor(spec.grid_extent[1] / spec.pixel_size[0])
            # iterate through the blocks by row then col
            ystep = spec.tile_size[0] - spec.tile_overlap[0]
            xstep = spec.tile_size[1] - spec.tile_overlap[1]
            for y in range(0, row_px, ystep):
                for x in range(0, col_px, xstep):
                    # dynamically adjust window size to stay within bounds
                    th = min(spec.tile_size[0], row_px - y) # at the last row
                    tw = min(spec.tile_size[1], col_px - x) # at the last col
                    # set up the window and update the result dict
                    window = rasterio.windows.Window(x, y, tw, th) # type: ignore
                    self._data[(x, y)] =  window
        elif self.mode == 'tiles':
            assert spec.grid_shape is not None
            # iterate through the blocks by row then col
            ystep = spec.tile_size[0] - spec.tile_overlap[0]
            xstep = spec.tile_size[1] - spec.tile_overlap[1]
            for row in range(spec.grid_shape[0]):
                for col in range(spec.grid_shape[1]):
                    tw = spec.tile_size[1]
                    th = spec.tile_size[0]
                    x = col * tw
                    y = row * th
                    window = rasterio.windows.Window(x, y, tw, th) # type: ignore
                    self._data[(x, y)] =  window
        else:
            raise ValueError('Invalid extent mode')

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class GridSpec:
    '''
    World grid specification.

    Conventions:
    - CRS space uses (x, y).
    - Pixel/array space uses (row, col).
    - All pixel sizes are positive magnitudes (x, y).
    - All shapes and tile sizes use (rows, cols).

    Note: this class is called during `GridLayout` construction, where
    the mode is determined.
    '''
    crs: str                                        # a projected CRS
    tile_size: tuple[int, int]                      # rows, cols in pixels
    tile_overlap: tuple[int, int]                   # rows, cols in pixels
    pixel_size: tuple[int, int]                     # xsize, ysize in CRS units
    origin: tuple[float, float]                     # x, y in CRS units
    grid_extent: tuple[float, float] | None = None  # H_y, W_x in in CRS units
    grid_shape: tuple[int, int] | None = None       # rows, cols as n of tiles

    def __post_init__(self):
        ts, to = self.tile_size, self.tile_overlap
        if not (to[0] < ts[0] and to[1] < ts[1]):
            raise ValueError('Overlap must be smaller than block size.')
