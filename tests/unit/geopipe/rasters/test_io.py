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

'''Unit tests for raster I/O context managers and DatasetInfo.'''

# standard imports
import pathlib
# third-party imports
import numpy
import pytest
import rasterio
# local imports
import landseg.geopipe.rasters as rasters


# ----- `read_raster` and `read_rasters` tests
def test_read_raster_success(dummy_geotiff_factory):
    '''
    Given: A valid GeoTIFF file.
    When: Opening it via `read_raster` context manager.
    Then: Yield opened reader instance and close it upon exit.
    '''
    r1 = dummy_geotiff_factory(
        filename='r1.tif', width=10, height=10, bands=1, dtype=numpy.uint8
    )

    with rasters.read_raster(r1) as reader:
        assert reader is not None
        assert reader.width == 10
        assert not reader.closed

    assert reader.closed


def test_read_raster_missing_raises():
    '''
    Given: A non-existent raster file path.
    When: Opening via `read_raster`.
    Then: Raise FileNotFoundError.
    '''
    missing_fp = pathlib.Path('/nonexistent/path/raster.tif')
    with pytest.raises(FileNotFoundError, match='Raster not found'):
        with rasters.read_raster(missing_fp):
            pass


def test_read_rasters_success(dummy_geotiff_factory):
    '''
    Given: Two valid GeoTIFF files and a None placeholder.
    When: Opening them via `read_rasters` context manager.
    Then: Yield opened reader instances and close them upon exit.
    '''
    r1 = dummy_geotiff_factory(
        filename='r1.tif', width=10, height=10, bands=1, dtype=numpy.uint8
    )
    r2 = dummy_geotiff_factory(
        filename='r2.tif', width=10, height=10, bands=1, dtype=numpy.uint8
    )

    with rasters.read_rasters(r1, None, r2) as (reader1, none_r, reader2):
        assert reader1 is not None
        assert none_r is None
        assert reader2 is not None
        assert reader1.width == 10
        assert reader2.width == 10
        assert not reader1.closed
        assert not reader2.closed

    assert reader1.closed
    assert reader2.closed


# ----- `DatasetInfo` and `get_info` tests
def test_get_info_and_band_map(dummy_geotiff_factory):
    '''
    Given: A multi-band GeoTIFF with band descriptions.
    When: Calling `get_info` and accessing `band_map`.
    Then: Return a `DatasetInfo` container with correct `band_map`.
    '''
    path = dummy_geotiff_factory(
        filename='test_bands.tif',
        width=8,
        height=8,
        bands=2,
    )
    with rasterio.open(path, 'r+') as src:
        src.set_band_description(1, 'RED')
        src.set_band_description(2, 'Green')

    info = rasters.get_info(path)
    assert info.width == 8
    assert info.height == 8
    assert info.count == 2
    assert info.band_map == {'red': 0, 'green': 1}


# ----- `read_window` tests
def test_read_window(dummy_geotiff_factory):
    '''
    Given: A GeoTIFF file.
    When: Calling `read_window` over an interior window.
    Then: Return numpy array with shape (bands, height, width).
    '''
    r = dummy_geotiff_factory(
        filename='win.tif', width=20, height=20, bands=3, dtype=numpy.uint8
    )
    with rasters.read_raster(r) as src:
        arr = rasters.read_window(src, rasters.Window(2, 2, 8, 8))
        assert arr.shape == (3, 8, 8)
