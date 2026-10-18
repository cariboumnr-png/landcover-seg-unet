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

# pylint: disable=duplicate-code

'''Unit tests for world grid builder (builder.py).'''

# standard imports
import dataclasses
# third-party imports
import pytest
# local imports
import landseg.geopipe.grid.builder as grid_builder


@dataclasses.dataclass
class _Params:
    tile_size: tuple[int, int] = (8, 8)
    tile_stride: tuple[int, int] = (4, 4)
    ref_fpath: str | None = None
    crs_string: str | None = None
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None


# ----- `build_grid` tests
def test_build_grid_ref(dummy_geotiff_factory):
    '''
    Given: A reference raster created on disk.
    When: `build_grid` is executed in `'ref'` mode.
    Then: Correctly parse reference bounds and resolutions, and
        construct the GridLayout.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref_raster.tif',
        width=16,
        height=16,
        bands=1,
        crs='EPSG:32617',
    )

    grid = grid_builder.build_grid(
        tile_size=(8, 8),
        tile_stride=(4, 4),
        mode='ref',
        ref_fpath=ref_path,
    )

    # 16x16 pixels image, tiles specs of 8x8 with 4 overlap.
    # step size = 8-4 = 4.
    # coordinates range in range(0, 16, 4) -> [0, 4, 8, 12] (4 steps)
    # total tiles = 4 * 4 = 16 tiles
    assert len(grid) == 16
    assert grid.crs == 'EPSG:32617'


def test_build_grid_manual():
    '''
    Given: Explicit grid geometries and coordinates.
    When: `build_grid` is executed in `'manual'` mode.
    Then: Create the GridLayout matching the provided spatial
        extent bounds.
    '''
    grid = grid_builder.build_grid(
        tile_size=(8, 8),
        tile_stride=(4, 4),
        mode='manual',
        manual_specs=grid_builder.GridConfigs(
            crs_string='EPSG:32617',
            origin=(500000.0, 5000000.0),
            pixel_size=(10.0, 10.0),
            extent_in_crs_units=(160.0, 160.0),
        ),
    )
    assert len(grid) == 16


def test_build_grid_invalid():
    '''
    Given: An invalid configuration mode.
    When: `build_grid` is executed.
    Then: Raise a ValueError.
    '''
    with pytest.raises(ValueError, match='Invalid extent mode'):
        grid_builder.build_grid(
            tile_size=(8, 8),
            tile_stride=(4, 4),
            mode='invalid',
        )
