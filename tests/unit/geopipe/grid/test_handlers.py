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

'''Unit tests for geopipe grid domain handlers (handlers.py).'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid.builder as builder
import landseg.geopipe.grid.handlers as handlers


# ----- `validate_grid_inputs` tests
def test_validate_grid_inputs_ref_success(dummy_geotiff_factory):
    '''
    Given: A valid reference raster and valid tile dimensions.
    When: `validate_grid_inputs` is executed in 'ref' mode.
    Then: Return constructed GridLayout instance.
    '''
    ref_fp = dummy_geotiff_factory(
        filename='ref.tif', width=32, height=32, crs='EPSG:32617'
    )
    grid = handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='ref',
        ref_fpath=pathlib.Path(ref_fp),
    )
    assert isinstance(grid, geo_core.GridLayout)
    assert len(grid) > 0


def test_validate_grid_inputs_ref_missing_file(tmp_path):
    '''
    Given: A nonexistent reference raster path.
    When: `validate_grid_inputs` is executed in 'ref' mode.
    Then: Raise a ValueError indicating invalid reference raster.
    '''
    missing_fp = tmp_path / 'missing.tif'
    with pytest.raises(ValueError, match='Invalid reference raster'):
        handlers.validate_grid_inputs(
            tile_size=(16, 16),
            tile_stride=(8, 8),
            mode='ref',
            ref_fpath=missing_fp,
        )


def test_validate_grid_inputs_manual_success():
    '''
    Given: Valid manual specifications and tile dimensions.
    When: `validate_grid_inputs` is executed in 'manual' mode.
    Then: Return constructed GridLayout instance.
    '''
    specs = builder.GridConfigs(
        crs_string='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        extent_in_crs_units=(200.0, 200.0),
    )
    grid = handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='manual',
        manual_specs=specs,
    )
    assert isinstance(grid, geo_core.GridLayout)
    assert len(grid) > 0


def test_validate_grid_inputs_manual_missing_specs():
    '''
    Given: Manual mode without manual specifications.
    When: `validate_grid_inputs` is executed.
    Then: Raise a ValueError.
    '''
    with pytest.raises(ValueError, match='Manual grid specs'):
        handlers.validate_grid_inputs(
            tile_size=(16, 16),
            tile_stride=(8, 8),
            mode='manual',
            manual_specs=None,
        )


def test_validate_grid_inputs_invalid_mode():
    '''
    Given: An unsupported grid mode string.
    When: `validate_grid_inputs` is executed.
    Then: Raise a ValueError.
    '''
    with pytest.raises(ValueError, match='Invalid extent mode'):
        handlers.validate_grid_inputs(
            tile_size=(16, 16),
            tile_stride=(8, 8),
            mode='invalid_mode',
        )
