# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

Public APIs:
    - `GridPayload`: TypedDict for serialized grid payload.
    - `GridMeta`: TypedDict for grid metadata.
    - `GridSpec`: Dataclass specifying world grid parameters.
    - `GridLayout`: Raster-agnostic grid layout of tile windows.
    - `get_grid_report_fpath`: Canonical path to grid report artifact.
    - `load_grid_from_fpath`: Load world grid layout directly from file.
    - `read_grid_report`: Read grid report JSON and extract summary.
'''

# standard imports
from __future__ import annotations
import collections.abc
import dataclasses
import math
import os
import typing
# third-party imports
import rasterio
import rasterio.crs
import rasterio.io
import rasterio.windows
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.grid as grid_contracts


# ----- typing aliases
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
'''Dataset reader handle for opened raster files.'''

RasterWindow: typing.TypeAlias = rasterio.windows.Window
'''Pixel window offset and slice for raster reads and writes.'''

RasterWindowDict: typing.TypeAlias = dict[tuple[int, int], RasterWindow]
'''Mapping of pixel-origin coordinates to raster windows.'''


# ----- public types
class GridPayload(typing.TypedDict):
    '''
    Serializable artifact for `GridLayout`.

    Fields:
        schema_id:
            Versioned identifier describing the serialization contract.
        artifact_meta:
            Lightweight metadata required to reconstruct grid layout.
        data:
            Serialized tile coordinates and raster window offsets.
    '''
    schema_id: str
    artifact_meta: GridMeta
    data: list[list[int]]


class GridMeta(typing.TypedDict):
    '''Lightweight metadata describing a `GridLayout` artifact.'''
    gid: str
    spec: dict[str, typing.Any]
    extent: tuple[int, int]


# ----- public dataclasses
@dataclasses.dataclass
class GridSpec:
    '''Specification for constructing a world grid.'''
    crs: str                                      # a projected CRS
    origin: tuple[float, float]                   # x, y in CRS units
    pixel_size: tuple[float, float]               # xsize, ysize in CRS units
    tile_size: tuple[int, int]                    # rows, cols in pixels
    tile_stride: tuple[int, int]                  # rows, cols in pixels
    grid_extent: tuple[float, float]              # H_y, W_x in in CRS units

    def __post_init__(self):
        '''Validate tile stride is smaller than tile size.'''
        ts, to = self.tile_size, self.tile_stride
        if not (to[0] < ts[0] and to[1] < ts[1]):
            raise ValueError('Overlap must be smaller than block size.')


# ----- public classes
class GridLayout(collections.abc.Mapping[tuple[int, int], RasterWindow]):
    '''
    Raster-agnostic grid layout represented as tile windows.

    A `GridLayout` defines a fixed tiling scheme over a projected CRS,
    producing a mapping from pixel-origin coordinates `(x_px, y_px)` to
    rasterio window objects.

    Schema:
        SCHEMA_ID = 'grid_layout_payload/v1'
    '''

    # current payload schema
    SCHEMA_ID: str = 'grid_layout_payload/v1'

    def __init__(self, spec: GridSpec):
        '''
        Initialize a `GridLayout` from a specification.

        Args:
            spec:
                Configuration object defining CRS, resolution, tile size
                stride, and grid extent.
        '''
        # ingest spec and init attributes
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
        '''Return a canonical identifier for the grid configuration.'''
        return self.generate_gid(self._spec.tile_size, self._spec.tile_stride)

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
        return self._spec.tile_stride

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

    @property
    def transform(self) -> rasterio.Affine:
        '''Return the affine transform for the grid.'''
        return rasterio.Affine(
            self._spec.pixel_size[0],
            0.0,
            self._spec.origin[0],
            0.0,
            -self._spec.pixel_size[1],
            self._spec.origin[1],
        )

    # ----- alternative constructor
    @classmethod
    def from_payload(cls, payload: GridPayload) -> GridLayout:
        '''
        Reconstruct a `GridLayout` from a serialized payload.

        Args:
            payload:
                Dictionary containing grid configuration and windows.

        Returns:
            GridLayout:
                A `GridLayout` instance with restored state.
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
        obj._extent = meta['extent']
        obj._data = parsed

        spec = dict(meta['spec'])
        for key in (
            'origin',
            'pixel_size',
            'tile_size',
            'tile_stride',
            'grid_extent'
        ):
            if spec.get(key) is not None:
                spec[key] = tuple(spec[key])
        obj._spec = GridSpec(**spec)

        # init offset (runtime attribute)
        obj._offset_px = (0, 0)
        # return class object
        return obj

    @classmethod
    def from_fpath(cls, fpath: str) -> GridLayout:
        '''
        Load a world grid layout directly from a serialized JSON file.

        Args:
            fpath:
                File path to the serialized grid JSON artifact.

        Returns:
            GridLayout:
                Restored GridLayout instance.
        '''
        ctrl = artifacts.PayloadController[
            list[list[int]], GridMeta
        ].load_or_fail(fpath, schema_id=cls.SCHEMA_ID)
        payload = ctrl.load()
        return cls.from_payload(payload)

    # ----- public method
    def offset_from(self, src: RasterReader | rasterio.Affine) -> None:
        '''
        Compute pixel offset to align the grid with a raster.

        Args:
            src:
                A raster dataset reader or affine transform describing
                the raster's spatial reference.
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
            GridPayload:
                Payload containing grid metadata and window definitions.
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
                'spec': dataclasses.asdict(self._spec),
                'extent': self._extent,
            },
            'data': canon
        }

    @staticmethod
    def generate_gid(
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int]
    ) -> str:
        '''
        Return a canonical identifier for the grid configuration.

        Args:
            tile_size:
                Tile dimensions in pixels as (rows, cols).
            tile_stride:
                Tile stride in pixels as (rows, cols).

        Returns:
            str:
                Canonical grid identifier string.
        '''
        row_size, col_size = tile_size
        row_stride, col_stride = tile_stride
        return f'grid_row_{row_size}_{row_stride}_col_{col_size}_{col_stride}'

    # ----- private method
    def _generate(self) -> None:
        '''Derive spatial extent from inputs and divide it into a grid.'''
        spec = self._spec
        # get extent dimensions (in crs units)
        assert spec.grid_extent is not None
        row_px = math.floor(spec.grid_extent[0] / spec.pixel_size[1])
        col_px = math.floor(spec.grid_extent[1] / spec.pixel_size[0])
        # iterate through the blocks by row then col
        ystep = spec.tile_size[0] - spec.tile_stride[0]
        xstep = spec.tile_size[1] - spec.tile_stride[1]
        for y in range(0, row_px, ystep):
            for x in range(0, col_px, xstep):
                # dynamically adjust window size to stay within bounds
                th = min(spec.tile_size[0], row_px - y) # at the last row
                tw = min(spec.tile_size[1], col_px - x) # at the last col
                # set up the window and update the result dict
                window = RasterWindow(x, y, tw, th) # type: ignore
                self._data[(x, y)] = window
        self._extent = row_px, col_px


# ----- public functions
def load_grid_from_fpath(fpath: str) -> GridLayout:
    '''
    Load a world grid layout directly from a file path.

    Args:
        fpath:
            File path to the serialized grid JSON artifact.

    Returns:
        GridLayout:
            Restored GridLayout instance.
    '''
    return GridLayout.from_fpath(fpath)


def get_grid_report_fpath(output_dpath: str) -> str:
    '''
    Return canonical file path of the world grid report artifact.

    Args:
        output_dpath:
            Output directory containing world grid artifacts.

    Returns:
        str:
            Full path to the grid_report.json artifact.
    '''
    return os.path.join(output_dpath, 'grid_report.json')


def read_grid_report(report_fpath: str) -> grid_contracts.WorldGridReport:
    '''
    Read a grid execution report and extract world grid summary.

    Args:
        report_fpath:
            File path to the grid report JSON artifact.

    Returns:
        grid_contracts.WorldGridReport:
            World grid summary report extracted from the artifact.
    '''
    ctrl = artifacts.Controller[
        grid_contracts.GridReportSchema
    ].load_json_or_fail(report_fpath)
    report = ctrl.fetch()
    return report['grid']
