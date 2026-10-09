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

# pylint: disable=protected-access

'''
Unit tests for data foundation configuration schemas.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.data.harmonziation as harm_sec
import landseg.configs.schema.data.ingestion as ing_sec
import landseg.configs.schema.data.world_grid as grid_sec


# ----- `GridSpecs` tests
def test_grid_parameters_validation():
    '''
    Given: `GridSpecs` instances with square or non-square dimensions.
    When: `GridSpecs.validate()` is invoked.
    Then: Accept valid square configs or raise ValueError for invalid.
    '''
    params = grid_sec.GridSpecs(
        tile_size=(256, 256),
        tile_stride=(0, 0),
    )
    params.validate()

    with pytest.raises(ValueError, match='Only square blocks are supported'):
        grid_sec.GridSpecs(tile_size=(256, 512)).validate()

    with pytest.raises(ValueError, match='Only equal row/column stride'):
        grid_sec.GridSpecs(
            tile_size=(256, 256),
            tile_stride=(10, 20),
        ).validate()

    with pytest.raises(ValueError, match='Block size must be positive'):
        grid_sec.GridSpecs(tile_size=(0, 0)).validate()

    with pytest.raises(
        ValueError, match='Block stride must be zero or positive'
    ):
        grid_sec.GridSpecs(
            tile_size=(256, 256),
            tile_stride=(-1, -1),
        ).validate()


# ----- `WorldGridConfig` tests
def test_grid_cfg_validation(tmp_path):
    '''
    Given: `WorldGridConfig` instances with valid or invalid parameters.
    When: `WorldGridConfig.validate()` is called.
    Then: Pass valid ref grid definitions and raise error for invalid.
    '''
    ref_file = tmp_path / 'ref.tif'
    ref_file.write_text('dummy')

    grid_ref = grid_sec.WorldGridConfig(
        mode='ref',
        params=grid_sec.GridSpecs(
            ref_fpath=str(ref_file),
            crs_string='EPSG:32617',
            tile_size=(256, 256),
            tile_stride=(0, 0),
        )
    )
    grid_ref.validate()
    assert grid_ref.tile_specs_tuple == (256, 256, 0, 0)

    # missing reference file
    grid_missing_ref = grid_sec.WorldGridConfig(
        mode='ref',
        params=grid_sec.GridSpecs(
            ref_fpath=str(tmp_path / 'non_existent.tif'),
            crs_string='EPSG:32617',
        )
    )
    with pytest.raises(base.ConfigValidationError):
        grid_missing_ref.validate()

    # manual mode validation
    grid_manual = grid_sec.WorldGridConfig(
        mode='manual',
        params=grid_sec.GridSpecs(
            crs_string='EPSG:32617',
            origin=(0.0, 0.0),
            pixel_size=(10.0, 10.0),
            extent_in_crs_units=(100.0, 100.0),
        )
    )
    grid_manual.validate()
    assert grid_manual.spatial_resolution == 10.0

    # invalid manual CRS
    grid_invalid_crs = grid_sec.WorldGridConfig(
        mode='manual',
        params=grid_sec.GridSpecs(
            crs_string='INVALID_CRS',
            origin=(0.0, 0.0),
            pixel_size=(10.0, 10.0),
            extent_in_crs_units=(100.0, 100.0),
        )
    )
    with pytest.raises(ValueError, match='Invalid CRS'):
        grid_invalid_crs.validate()


# ----- `Domains` tests
def test_domains_management():
    '''
    Given: Default `Domains` configuration object.
    When: `Domains.validate()` is called.
    Then: Validate threshold settings.
    '''
    domains = ing_sec.Domains(valid_threshold=0.7, target_variance=0.9)
    domains.validate()
    assert domains.valid_threshold == 0.7


# ----- `DataBlocks` & `DataIngestionConfig` tests
def test_datablocks_and_data_validation():
    '''
    Given: Valid `DataBlocks` instance.
    When: `DataBlocks.validate()` and `DataIngestionConfig.validate()` run.
    Then: Validate data config.
    '''
    blocks = ing_sec.DataBlocks()
    blocks.validate()

    df = ing_sec.DataIngestionConfig(datablocks=blocks)
    df.validate()


def test_datablocks_add_features_validation():
    '''
    Given: `DataBlocks` instances with valid/invalid topo & spectral.
    When: `DataBlocks.validate()` runs.
    Then: Accept valid settings or raise error.
    '''
    # valid configurations
    valid = ing_sec.DataBlocks(
        add_topo=['slope', 'tpi'],
        add_spectral=['ndvi', 'nbr']
    )
    valid.validate()

    # invalid topo type
    with pytest.raises(base.ConfigValidationError):
        ing_sec.DataBlocks(add_topo='invalid').validate()

    # invalid topo item type
    with pytest.raises(ValueError, match='Expected type'):
        ing_sec.DataBlocks(add_topo=[123]).validate()

    # invalid topo feature name
    with pytest.raises(ValueError, match='must be in the following'):
        ing_sec.DataBlocks(add_topo=['unknown']).validate()

    # invalid spectral type
    with pytest.raises(base.ConfigValidationError):
        ing_sec.DataBlocks(add_spectral='ndvi').validate()

    # invalid spectral item type
    with pytest.raises(ValueError, match='Expected type'):
        ing_sec.DataBlocks(add_spectral=[123]).validate()

    # invalid spectral index name
    with pytest.raises(ValueError, match='must be in the following'):
        ing_sec.DataBlocks(add_spectral=['unknown']).validate()


# ----- `DataHarmonizationConfig` tests
def test_harmonization_cfg_validation(tmp_path):
    '''
    Given: `DataHarmonizationConfig` instances with parameters.
    When: `DataHarmonizationConfig.validate()` is called.
    Then: Accept valid settings.
    '''
    manifest_file = tmp_path / 'manifest.json'
    manifest_file.write_text('dummy')

    h_cfg = harm_sec.DataHarmonizationConfig(
        dataset_manifest=str(manifest_file),
        resampling_continuous='bilinear',
        resampling_categorical='nearest',
    )
    h_cfg.validate()
    assert h_cfg.resampling_continuous == 'bilinear'
