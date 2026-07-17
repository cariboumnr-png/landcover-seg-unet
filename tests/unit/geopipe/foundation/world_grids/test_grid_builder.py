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

'''Unit tests for world grid builder (builder.py).'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.foundation.world_grids.builder as grid_builder


# ----- `build_grid` tests
def test_build_grid_ref(dummy_geotiff_factory):
    '''
    Given: A reference raster created on disk.
    When: `build_grid` is executed in `'ref'` mode.
    Then: Correctly parse reference bounds and resolutions, and
        construct the GridLayout.
    '''
    ref_path = str(dummy_geotiff_factory('ref_raster.tif', 16, 16, 1))

    # set up grid parameters for ref mode
    config = grid_builder.GridParameters(
        mode='ref',
        crs='EPSG:32617',
        ref_fpath=ref_path,
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(8, 8, 4, 4)
    )

    grid = grid_builder.build_grid(config)

    # 16x16 pixels image, tiles specs of 8x8 with 4 overlap.
    # step size = 8-4 = 4.
    # coordinates range in range(0, 16, 4) -> [0, 4, 8, 12] (4 steps)
    # total tiles = 4 * 4 = 16 tiles
    assert len(grid) == 16
    assert grid.crs == 'EPSG:32617'


def test_build_grid_aoi():
    '''
    Given: Explicit grid geometries and coordinates.
    When: `build_grid` is executed in `'aoi'` mode.
    Then: Create the GridLayout matching the provided spatial
        extent bounds.
    '''
    config = grid_builder.GridParameters(
        mode='aoi',
        crs='EPSG:32617',
        ref_fpath='',
        origin=(500000.0, 5000000.0),
        pixel_size=(10.0, 10.0),
        grid_extent=(160.0, 160.0), # 16x16 pixels
        grid_shape=None,
        tile_specs=(8, 8, 4, 4)
    )

    grid = grid_builder.build_grid(config)
    assert len(grid) == 16


def test_build_grid_tiles():
    '''
    Given: Explicit coordinates and a fixed tile shape count.
    When: `build_grid` is executed in `'tiles'` mode.
    Then: Construct a grid matching the specified row/col tile
        dimensions.
    '''
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

    grid = grid_builder.build_grid(config)
    assert len(grid) == 4


def test_build_grid_invalid():
    '''
    Given: An invalid configuration mode.
    When: `build_grid` is executed.
    Then: Raise a ValueError.
    '''
    config = grid_builder.GridParameters(
        mode='invalid', # type: ignore
        crs='EPSG:32617',
        ref_fpath='',
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(8, 8, 4, 4)
    )

    with pytest.raises(ValueError, match='Invalid extent mode'):
        grid_builder.build_grid(config)
