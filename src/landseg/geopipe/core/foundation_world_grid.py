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

# pylint: disable=c-extension-no-member
'''
World-grid tiling utilities.

This module defines a deterministic, raster-agnostic world grid used to
partition spatial data into stable, pixel-aligned tiles. It provides
structures and utilities for generating, indexing, and serializing grid
layouts independent of any specific raster dataset.

Key principles:
- The grid is immutable once defined and serves as a canonical spatial
  reference.
- Tile indices are stable across datasets and expressed in pixel-space
  coordinates relative to the grid origin.
- No raster reprojection, resampling, or alignment is performed here;
  all inputs must already be aligned upstream.

Typical usage:
- Define a `GridSpec` describing CRS, resolution, tile size, and extent.
- Construct a `GridLayout` to generate tile windows.
- Align the grid to a raster using `offset_from()` for consistent window
  extraction.

This module ensures reproducible tiling and indexing across datasets,
enabling consistent training, inference, and data integration workflows.
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
import rasterio.io
import rasterio.windows

# aliases
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
RasterWindow: typing.TypeAlias = rasterio.windows.Window
RasterWindowDict: typing.TypeAlias = dict[tuple[int, int], RasterWindow]

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class GridSpec:
    '''
    Specification for constructing a world grid.

    This dataclass defines the spatial reference, resolution, and tiling
    configuration used to generate a `GridLayout`.

    Conventions:
        - CRS coordinates use (x, y).
        - Pixel space uses (row, col).
        - Pixel sizes are positive magnitudes.
        - Tile sizes and overlaps are expressed in (rows, cols).

    Notes:
        Either `grid_extent` (for `'bbox'` mode) or `grid_shape`
        (for `'tiles'` mode) must be provided depending on how the grid
        is constructed.
    '''
    crs: str                                        # a projected CRS
    origin: tuple[float, float]                     # x, y in CRS units
    pixel_size: tuple[float, float]                 # xsize, ysize in CRS units
    tile_size: tuple[int, int]                      # rows, cols in pixels
    tile_overlap: tuple[int, int]                   # rows, cols in pixels
    grid_extent: tuple[float, float] | None = None  # H_y, W_x in in CRS units
    grid_shape: tuple[int, int] | None = None       # rows, cols as n of tiles

    def __post_init__(self):
        ts, to = self.tile_size, self.tile_overlap
        if not (to[0] < ts[0] and to[1] < ts[1]):
            raise ValueError('Overlap must be smaller than block size.')

# ---------------------------------Public Type---------------------------------
class GridPayload(typing.TypedDict):
    '''
    Serializable artifact for `GridLayout`.

    This follows the standard artifact contract:

    {
        "schema_id": str,
        "artifact_meta": `_GridMeta`,
        "data": `RasterWindowDict`
    }

    The grid layout is stored as a deterministic spatial index of raster
    windows. Metadata is separated from the raw window mapping to allow
    lightweight inspection without loading full spatial data.

    Fields:

    - **schema_id**:
        Versioned identifier describing the serialization contract.
    - **artifact_meta**:
        Lightweight descriptive metadata required to interpret and
        reconstruct the grid layout.
    - **data**:
        Mapping of tile coordinates to raster windows.
    '''
    schema_id: str
    artifact_meta: GridMeta
    data: list[list[int]]

class GridMeta(typing.TypedDict):
    '''Lightweight metadata describing a `GridLayout` artifact.'''
    gid: str
    mode: str
    spec: dict[str, typing.Any]
    extent: tuple[int, int]

# --------------------------------Public  Class--------------------------------
class GridLayout(collections.abc.Mapping[tuple[int, int], RasterWindow]):
    '''
    Raster-agnostic grid layout represented as tile windows.

    A `GridLayout` defines a fixed tiling scheme over a projected CRS,
    producing a mapping from pixel-origin coordinates `(x_px, y_px)` to
    rasterio window objects.

    Key features:
        - Stable indexing independent of any specific raster
        - Support for overlapping or non-overlapping tiles
        - Two construction modes: fixed bounding box or fixed tile count
        - Runtime alignment to rasters via pixel offsets

    The mapping behaves like a read-only dictionary where keys are
    pixel-origin coordinates and values are `rasterio.windows.Window`
    objects.

    Schema:
        SCHEMA_ID = 'grid_layout_payload/v1'
    '''

    # current payload schema
    SCHEMA_ID: str = 'grid_layout_payload/v1'

    def __init__(
        self,
        mode: typing.Literal['bbox', 'tiles'],
        spec: GridSpec
    ):
        '''
        Initialize a `GridLayout` from a specification.

        Args:
            mode:
                Grid construction mode:
                - `'bbox'`: derive tiles from a spatial extent
                - `'tiles'`: derive extent from a fixed tile grid

            spec:
                Configuration object defining CRS, resolution, tile size,
                overlap, and either extent or grid shape.

        Notes: The grid is generated immediately upon initialization.
        '''

        # ingest spec and init attributes
        self._mode = mode
        self._spec = spec
        self._extent: tuple[int, int] = (0, 0) # (rows, cols)
        self._data: RasterWindowDict = {}
        self._offset_px: tuple[int, int] = (0, 0)  # (dc_px, dr_px)
        # generate grid - self._data to be populated
        self._generate()

    # ----- container protocol
    def __getitem__(self, idx: tuple[int, int]) -> RasterWindow:
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
        return RasterWindow(xoff, yoff, width, height) # type: ignore

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

    # ----- property
    @property
    def gid(self) -> str:
        '''
        Return a canonical identifier for the grid configuration.

        Example:
            A grid with tile size of (H256, W256) and overlap of (H128,
            W128) will have a gid as `'grid_row_256_128_col_256_128'`.
        '''
        return (
            f'grid_row_{self._spec.tile_size[0]}_{self._spec.tile_overlap[0]}_'
            f'col_{self._spec.tile_size[1]}_{self._spec.tile_overlap[0]}'
        )

    @property
    def crs(self) -> str:
        '''Return the coordinate reference system of the grid.'''
        return self._spec.crs

    @property
    def origin(self) -> tuple[float, float]:
        '''Return the grid origin in CRS coordinates (x, y).'''
        return self._spec.origin

    @property
    def pixel_size(self) -> tuple[float, float]:
        '''Return the grid pixel size as (x, -y) in CRS units.'''
        return self._spec.pixel_size[0], -self._spec.pixel_size[1]

    @property
    def tile_size(self) -> tuple[int, int]:
        '''Return the tile size in pixels as (rows, cols).'''
        return self._spec.tile_size

    @property
    def tile_overlap(self) -> tuple[int, int]:
        '''Return the overlap between adjacent tiles in pixels.'''
        return self._spec.tile_overlap

    @property
    def extent(self) -> tuple[int, int]:
        '''Return the grid extent in pixels as (height, width).'''
        return self._extent

    @property
    def h(self) -> int:
        '''Return the grid height in pixels.'''
        return self._extent[0]

    @property
    def w(self) -> int:
        '''Return the grid width in pixels.'''
        return self._extent[1]

    # ----- public method
    def offset_from(self, src: RasterReader | rasterio.Affine) -> None:
        '''
        Compute pixel offset to align the grid with a raster.

        This method adjusts the grid so that its tile windows correctly
        map onto a raster with matching CRS and resolution.

        Args:
            src:
                A raster dataset reader or affine transform describing
                the raster's spatial reference.

        Notes: The raster must already be aligned in CRS and pixel size.
        Only integer pixel offsets are supported.
        '''

        # if a raster reader handler is provided:
        if isinstance(src, RasterReader):
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

    def to_payload(self) -> GridPayload:
        '''
        Convert the grid layout into a serializable payload.

        Returns:
            A `GridLayoutPayload` containing all necessary information
            to reconstruct the layout.
        '''

        # get canonical serialization of the data (JSON compatible)
        canon: list[list[int]] = []
        for k, w in sorted(self._data.items()):
            canon.append(
                [k[0], k[1], w.col_off, w.row_off, w.width, w.height]
            )

        return {
            'schema_id': self.SCHEMA_ID,
            'artifact_meta': {
                'gid': self.gid,
                'mode': self._mode,
                'spec': dataclasses.asdict(self._spec),
                'extent': self._extent,
            },
            'data': canon
        }

    @classmethod
    def from_payload(cls, payload: GridPayload) -> GridLayout:
        '''
        Reconstruct a `GridLayout` from a serialized payload.

        Args:
            payload:
                Dictionary containing grid configuration and windows.

        Returns:
            A `GridLayout` instance with restored state.

        Notes: Runtime attributes such as offsets are reset and must be
        recomputed if needed.
        '''

        # parse data from payload
        parsed: RasterWindowDict = {}
        for c in payload['data']:
            x, y, col_off, row_off, w, h = c
            window = RasterWindow(col_off, row_off, w, h) # type: ignore
            parsed[(x, y)] = window

        # create empty GridLayout instance
        obj = cls.__new__(cls)
        # populate attributes from payload
        meta = payload['artifact_meta']
        obj._mode = meta['mode']
        obj._spec = GridSpec(**meta['spec'])
        obj._extent = meta['extent']
        obj._data = parsed
        # init offset (runtime attribute)
        obj._offset_px = (0, 0)
        # return class object
        return obj

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
                    window = RasterWindow(x, y, tw, th) # type: ignore
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
                    window = RasterWindow(x, y, tw, th) # type: ignore
                    self._data[(x, y)] = window
            self._extent = (
                (spec.grid_shape[0] - 1) * ystep + spec.tile_size[0],
                (spec.grid_shape[1] - 1) * xstep + spec.tile_size[1]
            )
        else:
            raise ValueError('Invalid extent mode')
