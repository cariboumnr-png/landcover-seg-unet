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

# pylint: disable=protected-access

'''Unit tests for domain maps lifecycle management (lifecycle.py).'''

# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.domain_maps.lifecycle as domain_lifecycle


# ----- `prepare_domain_maps` tests
def test_prepare_domain_maps_build_and_load(tmp_path, mocker):
    '''
    Given: A grid, configs, and mock raster tiles.
    When: `prepare_domain_maps` is executed.
    Then: Build a new DomainTileMap, serialize it to disk, and
        load it on subsequent calls.
    '''
    grid_spec = geo_core.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        tile_size=(8, 8),
        tile_overlap=(0, 0),
        grid_shape=(1, 1)
    )
    grid = geo_core.GridLayout(mode='tiles', spec=grid_spec)

    config = domain_lifecycle.DomainBuildingParameters(
        input_fpath='dummy_input.tif',
        domain_fpath=str(tmp_path / 'domain_map.json'),
        tiles_fpath=str(tmp_path / 'tiles.npz'),
        index_base=0,
        valid_threshold=0.5,
        target_variance=0.9
    )

    # mock tiles to return when prepping mapping
    mock_tiles = {
        (-999, -999): numpy.array([[1]]),
        (0, 0): numpy.array([[0, 0], [0, 1]]),
        (0, 1): numpy.array([[1, 1], [1, 1]]),
    }
    mocker.patch(
        'landseg.geopipe.foundation.domain_maps.lifecycle._prep_mapping',
        return_value=mock_tiles
    )

    mock_logger = mocker.Mock()

    # execution 1: build
    domain_lifecycle.prepare_domain_maps(
        grid,
        [config],
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=mock_logger
    )

    assert (tmp_path / 'domain_map.json').exists()
    mock_logger.add_domain_report.assert_called_once()
    report1 = mock_logger.add_domain_report.call_args[0][0]
    assert report1['status'] == 'created'
    assert report1['stats']['valid_coords_count'] == 2

    mock_logger.reset_mock()

    # execution 2: load
    domain_lifecycle.prepare_domain_maps(
        grid,
        [config],
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=mock_logger
    )

    mock_logger.add_domain_report.assert_called_once()
    report2 = mock_logger.add_domain_report.call_args[0][0]
    assert report2['status'] == 'loaded'
