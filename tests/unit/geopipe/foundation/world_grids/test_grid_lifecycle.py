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

'''Unit tests for world grid lifecycle management (lifecycle.py).'''

# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.foundation.world_grids.builder as grid_builder
import landseg.geopipe.foundation.world_grids.lifecycle as grid_lifecycle


# ----- `prepare_world_grid` tests
def test_prepare_world_grid_build(tmp_path, mocker):
    '''
    Given: A clean temporary folder (no saved grid JSON exists).
    When: `prepare_world_grid` is executed.
    Then: Build a new GridLayout, save it, and log status as
        'created_and_loaded'.
    '''
    grid_fpath = str(tmp_path / 'world_grid.json')
    config = grid_builder.GridParameters(
        mode='tiles',
        crs='EPSG:32617',
        ref_fpath='',
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        grid_extent=None,
        grid_shape=(2, 2),
        tile_specs=(8, 8, 4, 4)
    )

    mock_logger = mocker.Mock()

    grid = grid_lifecycle.prepare_world_grid(
        grid_fpath=grid_fpath,
        config=config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=mock_logger
    )

    assert len(grid) == 4
    # verify that output JSON was saved to disk
    assert (tmp_path / 'world_grid.json').exists()

    # verify report call on mock logger
    mock_logger.set_world_grid_report.assert_called_once()
    report = mock_logger.set_world_grid_report.call_args[0][0]
    assert report['status'] == 'created_and_loaded'
    assert report['grid_id'] == grid.gid


def test_prepare_world_grid_load(tmp_path, mocker):
    '''
    Given: An already persisted grid JSON file on disk.
    When: `prepare_world_grid` is executed.
    Then: Load the grid from disk directly without rebuilding, and
        log status as 'loaded'.
    '''
    grid_fpath = str(tmp_path / 'world_grid.json')
    config = grid_builder.GridParameters(
        mode='tiles',
        crs='EPSG:32617',
        ref_fpath='',
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        grid_extent=None,
        grid_shape=(2, 2),
        tile_specs=(8, 8, 4, 4)
    )

    mock_logger = mocker.Mock()

    # build once to save to disk
    grid_lifecycle.prepare_world_grid(
        grid_fpath=grid_fpath,
        config=config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=mock_logger
    )
    mock_logger.reset_mock()

    # execution 2: load from disk
    grid = grid_lifecycle.prepare_world_grid(
        grid_fpath=grid_fpath,
        config=config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=mock_logger
    )

    assert len(grid) == 4
    mock_logger.set_world_grid_report.assert_called_once()
    report = mock_logger.set_world_grid_report.call_args[0][0]
    assert report['status'] == 'loaded'
