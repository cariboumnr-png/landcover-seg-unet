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
Unit tests for `landseg.configs.schema.sections.data`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.sections.data as data


# ----- `_GridParameters` tests
def test_grid_parameters_validation():
    '''
    Given: `_GridParameters` instances with square or non-square dimensions.
    When: `_GridParameters.validate()` is invoked.
    Then: Accept valid square configs or raise ValueError for invalid.
    '''
    params = data._GridParameters(
        tile_size=(256, 256),
        tile_stride=(0, 0),
    )
    params.validate()

    with pytest.raises(ValueError, match='Only square blocks are supported'):
        data._GridParameters(tile_size=(256, 512)).validate()

    with pytest.raises(ValueError, match='Only equal row/column stride'):
        data._GridParameters(
            tile_size=(256, 256),
            tile_stride=(10, 20),
        ).validate()

    with pytest.raises(ValueError, match='Block size must be positive'):
        data._GridParameters(tile_size=(0, 0)).validate()

    with pytest.raises(ValueError, match='Block stride must be zero or positive'):
        data._GridParameters(
            tile_size=(256, 256),
            tile_stride=(-1, -1),
        ).validate()


# ----- `_GridCfg` tests
def test_grid_cfg_validation(tmp_path):
    '''
    Given: `_GridCfg` instances with valid or invalid parameters.
    When: `_GridCfg.validate()` is called.
    Then: Pass valid ref grid definitions and raise error for invalid.
    '''
    ref_file = tmp_path / 'ref.tif'
    ref_file.write_text('dummy')

    grid_ref = data._GridCfg(
        mode='ref',
        params=data._GridParameters(
            ref_fpath=str(ref_file),
            crs_string='EPSG:32617',
            tile_size=(256, 256),
            tile_stride=(0, 0),
        )
    )
    grid_ref.validate()
    assert grid_ref.tile_specs_tuple == (256, 256, 0, 0)

    # missing reference file
    grid_missing_ref = data._GridCfg(
        mode='ref',
        params=data._GridParameters(
            ref_fpath=str(tmp_path / 'non_existent.tif'),
            crs_string='EPSG:32617',
        )
    )
    with pytest.raises(FileNotFoundError):
        grid_missing_ref.validate()

    # manual mode validation
    grid_manual = data._GridCfg(
        mode='manual',
        params=data._GridParameters(
            crs_string='EPSG:32617',
            origin=(0.0, 0.0),
            pixel_size=(10.0, 10.0),
            extent_in_crs_units=(100.0, 100.0),
        )
    )
    grid_manual.validate()
    assert grid_manual.spatial_resolution == 10.0

    # invalid manual CRS
    grid_invalid_crs = data._GridCfg(
        mode='manual',
        params=data._GridParameters(
            crs_string='INVALID_CRS',
            origin=(0.0, 0.0),
            pixel_size=(10.0, 10.0),
            extent_in_crs_units=(100.0, 100.0),
        )
    )
    with pytest.raises(ValueError, match='Invalid CRS'):
        grid_invalid_crs.validate()


# ----- `_Domains` tests
def test_domains_management():
    '''
    Given: Default `_Domains` configuration object.
    When: `_Domains.validate()` is called.
    Then: Validate threshold settings.
    '''
    domains = data._Domains(valid_threshold=0.7, target_variance=0.9)
    domains.validate()
    assert domains.valid_threshold == 0.7


# ----- `_DataBlocks` & `_IngestionCfg` tests
def test_datablocks_and_data_validation():
    '''
    Given: Valid `_DataBlocks` instance.
    When: `_DataBlocks.validate()` and `_IngestionCfg.validate()` run.
    Then: Validate data config.
    '''
    blocks = data._DataBlocks()
    blocks.validate()

    df = data._IngestionCfg(datablocks=blocks)
    df.validate()


def test_datablocks_add_features_validation():
    '''
    Given: `_DataBlocks` instances with valid/invalid topo & spectral.
    When: `_DataBlocks.validate()` runs.
    Then: Accept valid settings or raise TypeError / ValueError.
    '''
    # valid configurations
    valid = data._DataBlocks(add_topo=True, add_spectral=['ndvi', 'NBR'])
    valid.validate()

    # invalid topo type
    with pytest.raises(TypeError, match='add_topo must be a bool'):
        data._DataBlocks(add_topo='invalid').validate()

    # invalid spectral type
    with pytest.raises(TypeError, match='add_spectral must be a list'):
        data._DataBlocks(add_spectral='ndvi').validate()

    with pytest.raises(TypeError, match='Spectral index must be a string'):
        data._DataBlocks(add_spectral=[123]).validate()

    # invalid spectral index name
    with pytest.raises(ValueError, match='Invalid spectral index'):
        data._DataBlocks(add_spectral=['unknown']).validate()


# ----- `_HarmonizationCfg` tests
def test_harmonization_cfg_validation():
    '''
    Given: `_HarmonizationCfg` instances with parameters.
    When: `_HarmonizationCfg.validate()` is called.
    Then: Accept valid settings.
    '''
    h_cfg = data._HarmonizationCfg(
        dataset_manifest='/path/to/manifest.json',
        resampling_continuous='bilinear',
        resampling_categorical='nearest',
    )
    h_cfg.validate()
    assert h_cfg.resampling_continuous == 'bilinear'
