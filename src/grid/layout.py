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
import typing
# third-party imports
import rasterio
import rasterio.crs
# local imports
import alias

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
    pixel_size: tuple[float, float]                 # xsize, ysize in CRS units
    origin: tuple[float, float]                     # x, y in CRS units
    grid_extent: tuple[float, float] | None = None  # H_y, W_x in in CRS units
    grid_shape: tuple[int, int] | None = None       # rows, cols as n of tiles

    def __post_init__(self):
        ts, to = self.tile_size, self.tile_overlap
        if not (to[0] < ts[0] and to[1] < ts[1]):
            raise ValueError('Overlap must be smaller than block size.')

# ---------------------------------Public Type---------------------------------
class GridLayoutPayload(typing.TypedDict):
    '''GridLayout payload for controlled de-/serialization.'''
    mode: str
    spec: dict[str, typing.Any]
    extent: tuple[int, int]
    windows: alias.RasterWindowDict

# --------------------------------Public  Class--------------------------------
class GridLayout(collections.abc.Mapping[tuple[int, int], alias.RasterWindow]):
    '''
    Raster-agnostic grid layout as a dictionary of rasterio windows.

    **Key space**: dictionary keys are pixel-origin coordinates
    `(x_px, y_px)` relative to the *grid origin* (top-left). These are
    stable across rasters. Alignment to a specific raster is applied
    at access time; it does not change the keys.
    '''

    # current payload schema
    SCHEMA_ID: str = 'grid_layout_payload/v1'

    def __init__(
        self,
        mode: str,
        spec: GridSpec
    ):
        '''
        Initiate the `GridLayout` instance.

        Supports two modes:
        - `'bbox'`: the grid is bounded by origin x, y and rows, cols in
            pixels. Might contain ragged tiles at right and bottom edges.
            In this mode, `spec.grid_extent` must be provided.
        - `'tiles'`: the grid is bounded by origin x, y and number of
            tiles along row and col.
            In this mode, `spec.grid_shape` must be provided.

        See `GridSpec` for details in configuration.

        Args:
            mode: Grid generation mode (`'bbox'` or `'tiles'`).
            spec: A dataclass with layout configurations.
        '''

        # ingest spec and init attributes
        self._mode = mode
        self._spec = spec
        self._extent: tuple[int, int] = (0, 0) # (rows, cols)
        self._data: alias.RasterWindowDict = {}
        self._offset_px: tuple[int, int] = (0, 0)  # (dc_px, dr_px)
        # generate grid - self._data to be populated
        self._generate()

    # ----- container protocol
    def __getitem__(self, idx: tuple[int, int]) -> alias.RasterWindow:
        # fail fast on idx type check
        if (not isinstance(idx, tuple) or len(idx) != 2
            or not all(isinstance(v, int) for v in idx)):
            raise TypeError('Index must be (x, y) in pixels as integers.')
        # get base window
        base = self._data[idx]
        # get right/down offset (in pixels)
        dx, dy = self._offset_px
        # get components for raster window
        xoff, yoff = base.col_off - dx, base.row_off - dy
        width, height = base.width, base.height
        return alias.RasterWindow(xoff, yoff, width, height) # type: ignore

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ----- representation
    def __str__(self) -> str:
        return '\n'.join([
            'World grid details:',
            f'CRS: {self.crs}',
            f'Origin (x, y): {self.origin[0]:.4f}, {self.origin[1]:.4f}',
            f'Pixel size (x, -y): {self.pixel_size[0], self.pixel_size[1]}',
            f'Extent (height_px, width_px): {self.extent[0]}, {self.extent[1]}'
        ])

    # ----- public method
    def offset_from(self, src: alias.RasterReader | rasterio.Affine) -> None:
        '''
        Align the grid to a raster by computing an integer pixel offset.

        The raster must share the same CRS and pixel size as the grid.
        The offset is computed from the difference between grid origin
        and raster upper-left corner, expressed in pixels.

        After alignment, grid tile indices map consistently to raster
        pixel windows.

        Args:
            src: Input raster reader handler or an Affine transform. The
                raster must align with the grid's CRS and pixel size. If
                Affine transform is provided, please make sure the CRSs
                are aligned.
        '''

        # if a raster reader handler is provided:
        if isinstance(src, alias.RasterReader):
            # check target raster CRS
            grid_crs = rasterio.crs.CRS.from_user_input(self.crs)
            inpt_crs = rasterio.crs.CRS.from_user_input(src.crs)
            assert grid_crs == inpt_crs
            transform = src.transform
        # else src is already an Affine transform
        else:
            transform = src
        # get raster origin in CRS units
        rx, ry = transform.c, transform.f
        # get raster pixel size and check alignment with the grid
        res_x, res_y = transform.a, abs(transform.e)
        assert abs(self._spec.pixel_size[0] - res_x) < 1e-9
        assert abs(self._spec.pixel_size[1] - res_y) < 1e-9
        # get world grid origin in CRS units
        gx, gy = self._spec.origin
        # calculate origin offset in pixel
        dc = math.floor((rx - gx) / res_x)      # + right
        dr = math.floor((gy - ry) / res_y)      # + down
        self._offset_px = (dc, dr)

    def to_payload(self) -> GridLayoutPayload:
        '''Generate class payload for serialization.'''

        return {
            'mode': self._mode,
            'spec': dataclasses.asdict(self._spec),
            'extent': self._extent,
            'windows': self._data
        }

    @classmethod
    def from_payload(cls, payload: GridLayoutPayload) -> GridLayout:
        '''Instantiate the class with input payload.'''

        # create empty GridLayout instance
        obj = cls.__new__(cls)
        # populate attributes from payload
        obj._mode = payload['mode']
        obj._spec = GridSpec(**payload['spec'])
        obj._data = payload['windows']
        obj._extent = payload['extent']
        # init offset (runtime attribute)
        obj._offset_px = (0, 0)
        # return class object
        return obj

    # ----- property
    @property
    def crs(self) -> str:
        '''CRS of the layout.'''
        return self._spec.crs

    @property
    def origin(self) -> tuple[float, float]:
        '''Grid origin (x, y) in CRS units.'''
        return self._spec.origin

    @property
    def pixel_size(self) -> tuple[float, float]:
        '''Grid pixel size (x, -y) in CRS units.'''
        return self._spec.pixel_size[0], -self._spec.pixel_size[1]

    @property
    def tile_size(self) -> tuple[int, int]:
        '''Grid tile size (row, col) in pixels.'''
        return self._spec.tile_size

    @property
    def tile_overlap(self) -> tuple[int, int]:
        '''Grid tile overlap (row, col) in pixels.'''
        return self._spec.tile_overlap

    @property
    def extent(self) -> tuple[int, int]:
        '''Grid extent (height, width) in pixels.'''
        return self._extent

    @property
    def h(self) -> int:
        '''Grid extent height in pixels.'''
        return self._extent[0]

    @property
    def w(self) -> int:
        '''Grid extent width in pixels.'''
        return self._extent[1]

    # ----- private method
    def _generate(self) -> None:
        '''
        Derive spatial extent from inputs and divide it into a grid.

        Intended to work with a projected CRS with meter unit.
        '''

        spec = self._spec
        # get extent dimensions (in crs units)
        if self._mode == 'bbox':
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
                    window = alias.RasterWindow(x, y, tw, th) # type: ignore
                    self._data[(x, y)] = window
            self._extent = row_px, col_px
        elif self._mode == 'tiles':
            assert spec.grid_shape is not None
            # iterate through the blocks by row then col
            ystep = spec.tile_size[0] - spec.tile_overlap[0]
            xstep = spec.tile_size[1] - spec.tile_overlap[1]
            for row in range(spec.grid_shape[0]):
                for col in range(spec.grid_shape[1]):
                    tw = spec.tile_size[1]
                    th = spec.tile_size[0]
                    x = col * xstep
                    y = row * ystep
                    window = alias.RasterWindow(x, y, tw, th) # type: ignore
                    self._data[(x, y)] = window
            self._extent = (
                (spec.grid_shape[0] - 1) * ystep + spec.tile_size[0],
                (spec.grid_shape[1] - 1) * xstep + spec.tile_size[1]
            )
        else:
            raise ValueError('Invalid extent mode')
