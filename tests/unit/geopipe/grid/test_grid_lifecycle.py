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

'''Unit tests for world grid lifecycle management (lifecycle.py).'''

# standard imports
import dataclasses
# local imports
import landseg.geopipe.grid.lifecycle as grid_lifecycle


@dataclasses.dataclass
class _Params:
    tile_size: tuple[int, int] = (8, 8)
    tile_stride: tuple[int, int] = (4, 4)
    ref_fpath: str | None = None
    crs_string: str | None = None
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None


@dataclasses.dataclass
class _WorldGridConfig:
    mode: str
    params: _Params
    output_dpath: str


# ----- `prepare_world_grid` tests
def test_prepare_world_grid_build(tmp_path):
    '''
    Given: A clean temporary folder (no saved grid JSON exists).
    When: `prepare_world_grid` is executed.
    Then: Build a new GridLayout, save it, and return loaded=False.
    '''
    params = _Params(
        crs_string='EPSG:32617',
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        extent_in_crs_units=(160.0, 160.0),
        tile_size=(8, 8),
        tile_stride=(4, 4)
    )
    config = _WorldGridConfig(
        mode='manual',
        params=params,
        output_dpath=str(tmp_path)
    )

    is_loaded, grid_fp, world_grid = grid_lifecycle.prepare_world_grid(config)

    assert is_loaded is False
    assert len(world_grid) == 16
    assert (tmp_path / f'{world_grid.gid}.json').exists()
    assert grid_fp == str(tmp_path / f'{world_grid.gid}.json')


def test_prepare_world_grid_load(tmp_path):
    '''
    Given: An already persisted grid JSON file on disk.
    When: `prepare_world_grid` is executed.
    Then: Load the grid from disk directly and return loaded=True.
    '''
    params = _Params(
        crs_string='EPSG:32617',
        origin=(0.0, 0.0),
        pixel_size=(10.0, 10.0),
        extent_in_crs_units=(160.0, 160.0),
        tile_size=(8, 8),
        tile_stride=(4, 4)
    )
    config = _WorldGridConfig(
        mode='manual',
        params=params,
        output_dpath=str(tmp_path)
    )

    # build once to save to disk
    is_loaded_1, _, grid_1 = grid_lifecycle.prepare_world_grid(config)
    assert is_loaded_1 is False

    # execution 2: load from disk
    is_loaded_2, grid_fp_2, grid_2 = grid_lifecycle.prepare_world_grid(config)

    assert is_loaded_2 is True
    assert len(grid_2) == 16
    assert grid_2.gid == grid_1.gid
    assert grid_fp_2 == str(tmp_path / f'{grid_1.gid}.json')
