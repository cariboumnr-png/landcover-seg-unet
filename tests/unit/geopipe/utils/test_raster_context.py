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

'''Unit tests for raster open context manager (raster_context.py).'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.utils.raster_context as raster_context


# ----- `open_rasters` tests
def test_open_rasters_success(mocker):
    '''
    Given: A mixture of valid raster file paths and None entries.
    When: `open_rasters` is executed.
    Then: Yields opened readers for paths and None for None entries,
        closing all on exit.
    '''
    # mock os.path.exists to confirm existence
    mocker.patch('os.path.exists', return_value=True)

    # mock rasterio.open context manager behavior
    mock_reader = mocker.Mock()
    mock_open = mocker.patch('rasterio.open')
    mock_open.return_value.__enter__.return_value = mock_reader

    with raster_context.open_rasters(
        'raster1.tif', None, 'raster2.tif'
    ) as readers:
        assert len(readers) == 3
        assert readers[0] == mock_reader
        assert readers[1] is None
        assert readers[2] == mock_reader

    # check entering and exit behavior
    assert mock_open.call_count == 2
    assert mock_open.return_value.__exit__.call_count == 2


def test_open_rasters_missing_file(mocker):
    '''
    Given: A raster file path that does not exist.
    When: `open_rasters` is executed.
    Then: Raise an AssertionError.
    '''
    mocker.patch('os.path.exists', return_value=False)

    with pytest.raises(AssertionError, match='Raster not found'):
        with raster_context.open_rasters('missing.tif') as _:
            pass
