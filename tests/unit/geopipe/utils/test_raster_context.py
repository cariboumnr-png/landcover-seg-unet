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

'''Unit tests for open_rasters context manager (raster_context.py).'''

# standard imports
import pathlib
# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.utils.raster_context as raster_context


# ----- `open_rasters` tests
def test_open_rasters_success(dummy_geotiff_factory):
    '''
    Given: Two valid GeoTIFF files and a None placeholder.
    When: Opening them via `open_rasters` context manager.
    Then: Yield opened reader instances and close them upon exit.
    '''
    r1 = dummy_geotiff_factory(
        filename='r1.tif', width=10, height=10, bands=1, dtype=numpy.uint8
    )
    r2 = dummy_geotiff_factory(
        filename='r2.tif', width=10, height=10, bands=1, dtype=numpy.uint8
    )

    with raster_context.open_rasters(r1, None, r2) as (reader1, none_r, reader2):
        assert reader1 is not None
        assert none_r is None
        assert reader2 is not None
        assert reader1.width == 10
        assert reader2.width == 10
        assert not reader1.closed
        assert not reader2.closed

    assert reader1.closed
    assert reader2.closed


def test_open_rasters_missing_file_raises():
    '''
    Given: A non-existent raster file path.
    When: Opening via `open_rasters`.
    Then: Raise AssertionError with file not found message.
    '''
    missing_fp = pathlib.Path('/nonexistent/path/raster.tif')
    with pytest.raises(AssertionError, match='Raster not found'):
        with raster_context.open_rasters(missing_fp):
            pass


def test_open_rasters_empty():
    '''
    Given: No raster arguments.
    When: Opening via `open_rasters`.
    Then: Yield an empty tuple without error.
    '''
    with raster_context.open_rasters() as readers:
        assert not readers
