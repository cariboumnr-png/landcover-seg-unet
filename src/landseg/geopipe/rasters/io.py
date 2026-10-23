# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Context manager utility for opening raster datasets.

This module provides utilities to open multiple rasterio datasets safely
within a managed exit stack context.

Public APIs:
    - `open_rasters`: Context manager yielding opened raster readers.
'''

# standard imports
import dataclasses
import contextlib
import pathlib
import typing
# third-party imports
import numpy
import numpy.typing
import rasterio
import rasterio.coords
import rasterio.crs
import rasterio.transform
# local imports
import landseg.geopipe.rasters._types as t


@dataclasses.dataclass(frozen=True)
class DatasetInfo: # pylint: disable=too-many-instance-attributes
    '''Lightweight raster information container.'''
    crs: rasterio.crs.CRS | None
    driver: str
    transform: rasterio.transform.Affine
    width: int
    height: int
    bounds: rasterio.coords.BoundingBox
    res: tuple[float, float]
    nodata: float | int | None
    count: int
    dtypes: tuple[str, ...]
    descriptions: tuple[str | None, ...]
    tags: dict[str, str]
    band_tags: tuple[dict[str, str], ...]


# ----- public functions
def get_info(source: pathlib.Path | t.DatasetReader) -> DatasetInfo:
    '''Retrive raster info from a path or a `DatasetReader`.'''
    if isinstance(source, t.DatasetReader):
        return _get_info(source)
    with read_raster(source) as reader:
        return _get_info(reader)


@contextlib.contextmanager
def read_raster(
    path: str | pathlib.Path
) -> typing.Iterator[t.DatasetReader]:
    '''
    Read a raster safely and yield the dataset.

    Args:
        path:
            Path to target raster.

    Yields:
        RasterDataset:
            Opened dataset.
    '''
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f'Raster not found: {path}')
    with rasterio.open(path) as dataset:
        yield dataset


@contextlib.contextmanager
def read_rasters(
    *paths: str | pathlib.Path | None
) -> typing.Iterator[tuple[t.DatasetReader | None, ...]]:
    '''
    Read multiple rasters safely and yield a tuple of dataset readers.

    Args:
        *paths:
            Variable number of raster file paths or None values.

    Yields:
        tuple[alias.RasterReader | None, ...]:
            Tuple of opened dataset readers or None.
    '''
    with contextlib.ExitStack() as stack:
        opened_rasters: list[t.DatasetReader | None] = []

        for raster in paths:
            if raster is not None:
                if not pathlib.Path(raster).exists():
                    raise FileNotFoundError(f'Raster not found: {raster}')
                opened_raster = stack.enter_context(rasterio.open(raster))
                opened_rasters.append(opened_raster)
            else:
                opened_rasters.append(None)

        yield tuple(opened_rasters)


def read_window(
    dataset: t.DatasetReader,
    window: t.Window,
    *,
    indexes: int | list[int] | None = None,
    edge_mode: typing.Literal['boundless', 'reflect'] = 'boundless',
    padding: int = 0,
) -> numpy.typing.NDArray:
    '''
    Read a raster band for a given window with optional padding.

    The requested window is expanded by ``padding`` pixels on all sides,
    clipped to the dataset extent, and read from the source raster. If the
    padded window extends beyond the raster bounds, the missing pixels are
    filled using ``numpy.pad(..., mode="reflect")``.

    Args:
        dataset:
            Open raster dataset.
        window:
            Source window in pixel coordinates.
        indexes:
            Band index or indices to read using rasterio semantics.
            ``None`` reads all bands.
        edge_mode:
            Strategy used to handle reads that extend beyond the dataset
            boundary.
            - ``'boundless'``: use rasterio/GDAL boundless reads.
            - ``'reflect'``: clip to valid pixels and reflect-pad any
            out-of-bounds regions.
        padding:
            Number of context pixels added to each side of the requested
            window when ``edge_mode='reflect'``.

    Returns:
        numpy.ndarray:
            Raster data with any requested padding applied. Shape is
            ``(rows, cols)`` when reading a single band and
            ``(bands, rows, cols)`` when reading multiple bands.

    Raises:
        ValueError:
            If the computed padded window is invalid.
    '''
    if edge_mode == 'boundless':
        return _read_bands(dataset, indexes, window, boundless=True)

    if edge_mode == 'reflect':
        canvas_w, canvas_h = dataset.width, dataset.height

        # expand window within the original raster
        nw_x = max(window.col_off - padding, 0)
        nw_y = max(window.row_off - padding, 0)
        se_x = min(window.col_off + window.width + padding, canvas_w)
        se_y = min(window.row_off + window.height + padding, canvas_h)
        try:
            _window = t.Window(nw_x, nw_y, se_x - nw_x, se_y - nw_y) # type: ignore
        except ValueError as e:
            raise ValueError(
                f'Error reading DEM with pad ({padding}), padded raster window: '
                f'{nw_x}, {nw_y}, {se_x - nw_x}, {se_y - nw_y}'
            ) from e

        # get required padding on each side - no padding if within raster bound
        pad_width = (
            (
                max(0, padding - window.row_off),
                max(0, (window.row_off + window.height + padding) - canvas_h),
            ),
            (
                max(0, padding - window.col_off),
                max(0, (window.col_off + window.width + padding) - canvas_w),
            ),
        ) # ((top, bottom), (left, right))

        # raw array read to type, [H, W] or [n, H, W]
        arr = _read_bands(dataset, indexes, _window)

        if arr.ndim == 2:
            return numpy.pad(arr, pad_width, mode='reflect')

        return numpy.pad(arr, ((0, 0), *pad_width), mode='reflect')

    raise ValueError(f'Invalid edge mode: {edge_mode}')


# ----- private helpers
def _get_info(reader: t.DatasetReader) -> DatasetInfo:
    return DatasetInfo(
        crs=reader.crs,
        driver=reader.driver,
        transform=reader.transform,
        width=reader.width,
        height=reader.height,
        bounds=reader.bounds,
        res=reader.res,
        nodata=reader.nodata,
        count=reader.count,
        dtypes=reader.dtypes,
        descriptions=reader.descriptions,
        tags=reader.tags(),
        band_tags=tuple(reader.tags(i) for i in reader.indexes),
    )


def _read_bands(
    dataset: t.DatasetReader,
    indexes: int | list[int] | None,
    window: t.Window,
    *,
    boundless: bool = False,
) -> numpy.typing.NDArray:
    info = _get_info(dataset)
    if isinstance(indexes, int): # single band
        return dataset.read(indexes, window=window, boundless=boundless)
    bands = indexes or list(range(1, info.count + 1))
    # handle mixed dtypes in bands (rasterio will throw)
    to_stack = [
        dataset.read(b, window=window, boundless=boundless)
        for b in bands
    ]
    return numpy.stack(to_stack, axis=0)
