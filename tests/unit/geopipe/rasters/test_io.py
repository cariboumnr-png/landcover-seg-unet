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


def test_get_info_from_reader(dummy_geotiff_factory):
    '''
    Given: An already-opened DatasetReader instance.
    When: Passing the reader to `get_info`.
    Then: Return a `DatasetInfo` without re-opening from disk.
    '''
    path = dummy_geotiff_factory(
        filename='reader_info.tif',
        width=12,
        height=14,
        bands=1,
    )
    with rasters.read_raster(path) as reader:
        info = rasters.get_info(reader)
        assert info.width == 12
        assert info.height == 14
        assert info.count == 1


def test_band_map_fallback_and_duplicates(dummy_geotiff_factory):
    '''
    Given: Rasters with empty descriptions or duplicate descriptions.
    When: Accessing `band_map` property.
    Then: Fall back to band_N names or return empty dict on duplicates.
    '''
    # empty descriptions fallback
    p1 = dummy_geotiff_factory(
        filename='no_desc.tif',
        width=5,
        height=5,
        bands=2,
    )
    info1 = rasters.get_info(p1)
    assert info1.band_map == {'band_1': 0, 'band_2': 1}

    # duplicate descriptions return empty dict
    p2 = dummy_geotiff_factory(
        filename='dup_desc.tif',
        width=5,
        height=5,
        bands=2,
    )
    with rasterio.open(p2, 'r+') as src:
        src.set_band_description(1, 'BLUE')
        src.set_band_description(2, 'BLUE')

    info2 = rasters.get_info(p2)
    assert info2.band_map == {}


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
        arr = rasters.read_window(src, rasters.Window(2, 2, 8, 8)) # type: ignore
        assert arr.shape == (3, 8, 8)


def test_read_window_single_band_index(dummy_geotiff_factory):
    '''
    Given: A multi-band GeoTIFF file.
    When: Calling `read_window` with a single integer band index.
    Then: Return 2D numpy array of shape (height, width).
    '''
    r = dummy_geotiff_factory(
        filename='win_single.tif',
        width=20,
        height=20,
        bands=3,
        dtype=numpy.uint8,
    )
    with rasters.read_raster(r) as src:
        arr = rasters.read_window(
            src,
            rasters.Window(2, 2, 8, 8), # type: ignore
            indexes=1
        )
        assert arr.shape == (8, 8)


def test_read_window_reflect_padding(dummy_geotiff_factory):
    '''
    Given: A GeoTIFF file and a boundary window requiring padding.
    When: Calling `read_window` with `edge_mode='reflect'`.
    Then: Return padded array with expected dimensions.
    '''
    r = dummy_geotiff_factory(
        filename='win_pad.tif',
        width=20,
        height=20,
        bands=2,
        dtype=numpy.float32,
    )
    with rasters.read_raster(r) as src:
        # window at top-left corner (0, 0) with padding=2
        arr = rasters.read_window(
            src,
            rasters.Window(0, 0, 5, 5), # type: ignore
            padding=2,
            edge_mode='reflect',
        )
        assert arr.shape == (2, 9, 9)

        # single-band 2D reflect padding
        arr_single = rasters.read_window(
            src,
            rasters.Window(0, 0, 5, 5), # type: ignore
            indexes=1,
            padding=2,
            edge_mode='reflect',
        )
        assert arr_single.shape == (9, 9)


def test_read_window_invalid_edge_mode_raises(dummy_geotiff_factory):
    '''
    Given: A GeoTIFF file.
    When: Calling `read_window` with an unsupported edge mode.
    Then: Raise ValueError.
    '''
    r = dummy_geotiff_factory(
        filename='win_err.tif', width=10, height=10, bands=1
    )
    with rasters.read_raster(r) as src:
        with pytest.raises(ValueError, match='Invalid edge mode'):
            rasters.read_window(
                src,
                rasters.Window(0, 0, 5, 5), # type: ignore
                edge_mode='unsupported' # type: ignore
            )
