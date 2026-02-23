'''Context managers.'''

# standard imports
import contextlib
import os
import typing
# third-party imports
import rasterio
import rasterio.io

@contextlib.contextmanager
def open_rasters(
        *rasters: str | None
    ) -> typing.Iterator[tuple[rasterio.io.DatasetReader | None, ...]]:
    '''
    Open multiple rasters safely and yield a tuple of `DatasetReader`.

    Accepts any number of filepaths (or None). Existing paths are opened
    via rasterio, None values are preserved, and all files are closed
    automatically on exit.
    '''

    with contextlib.ExitStack() as stack:
        opened_rasters: list[rasterio.io.DatasetReader | None] = []

        for raster in rasters:
            if isinstance(raster, str):
                assert os.path.exists(raster), f'Raster not found: {raster}'
                opened_raster = stack.enter_context(rasterio.open(raster))
                opened_rasters.append(opened_raster)
            else:
                opened_rasters.append(None)

        yield tuple(opened_rasters)
