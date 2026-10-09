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
Unit tests for `landseg.configs.schema.root`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.data as data_schema
import landseg.configs.schema.data.ingestion as ingestion_sec
import landseg.configs.schema.models as models_schema
import landseg.configs.schema.root as root_mod
import landseg.configs.schema.session as session_schema
import landseg.configs.schema.study as study_schema


# ----- `RootConfig` tests
def test_root_config_defaults_and_as_dict():
    '''
    Given: Default `RootConfig` instantiation parameters.
    When: Instantiating `RootConfig` and calling `.as_dict`.
    Then: Initialize sub-sections and serialize config to dictionary.
    '''
    root = root_mod.RootConfig()

    assert isinstance(root.execution, root_mod.ExecutionContext)
    assert isinstance(root.data, data_schema.DataConfig)
    assert isinstance(root.models, models_schema.ModelsConfig)
    assert isinstance(root.session, session_schema.SessionConfig)
    assert isinstance(root.study, study_schema.StudyConfig)
    assert root.command == 'default'

    # dictionary serialization test
    cfg_dict = root.as_dict
    assert isinstance(cfg_dict, dict)
    assert 'execution' in cfg_dict
    assert 'data' in cfg_dict
    assert 'models' in cfg_dict


def test_root_config_command_validation():
    '''
    Given: A `RootConfig` instance with invalid command string.
    When: `RootConfig.validate()` is called.
    Then: Raise ConfigValidationError for unsupported commands.
    '''
    root = root_mod.RootConfig(command='unsupported_command')
    with pytest.raises(base.ConfigValidationError):
        root.validate()


def test_root_config_validate(tmp_path):
    '''
    Given: A `RootConfig` with valid foundation files and session.
    When: `RootConfig.validate()` is executed.
    Then: Complete validation across all configuration sub-sections.
    '''
    cfg_json = tmp_path / 'cfg.json'
    cfg_json.write_text('data')

    ref_tif = tmp_path / 'ref.tif'
    ref_tif.write_text('data')

    root = root_mod.RootConfig()
    root.data.world_grid.mode = 'ref'
    root.data.world_grid.params.ref_fpath = str(ref_tif)
    root.data.world_grid.params.crs_string = 'EPSG:32617'
    root.data.harmonization.dataset_manifest = str(cfg_json)
    root.session.orchestration.single_phase.num_epochs = 10

    root.validate()


def test_ingestion_config_harmonization_run_validation():
    '''
    Given: A `DataIngestionConfig` instance.
    When: Setting valid and invalid `harmonization_run` values.
    Then: Accept valid int/str values and raise error on invalid.
    '''
    cfg = ingestion_sec.DataIngestionConfig()

    # valid int, str, path, None
    cfg.harmonization_run = None
    cfg.validate()

    cfg.harmonization_run = 1
    cfg.validate()

    cfg.harmonization_run = 'run_0001'
    cfg.validate()

    cfg.harmonization_run = '/abs/path/run_0001'
    cfg.validate()

    # invalid non-positive int
    cfg.harmonization_run = -1
    with pytest.raises(base.ConfigValidationError):
        cfg.validate()

    # invalid type
    cfg.harmonization_run = [1] # type: ignore
    with pytest.raises(base.ConfigValidationError):
        cfg.validate()
