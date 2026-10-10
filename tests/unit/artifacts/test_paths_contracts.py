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
Unit tests for `landseg.artifacts.paths.contracts`.
'''

# third-party imports
import pytest
# local imports
import landseg.artifacts.paths as paths
import landseg.artifacts.paths.contracts as contracts
import landseg.configs as configs


# ----- `WorldGridIO` tests
def test_world_grid_io():
    '''
    Given: A target output path for world grid.
    When: Instantiating `WorldGridIO`.
    Then: Hold output `WorldGridPaths` instance.
    '''
    grid_paths = paths.WorldGridPaths('/custom/grid')
    io = contracts.WorldGridIO(output=grid_paths)
    assert io.output.root == '/custom/grid'


def test_world_grid_io_from_config():
    '''
    Given: A `RootConfig` with configured `world_grid.output_dpath`.
    When: Instantiating `WorldGridIO.from_config`.
    Then: Correctly resolve output `WorldGridPaths`.
    '''
    cfg = configs.RootConfig()
    cfg.data.world_grid.output_dpath = '/custom/grid'
    io = contracts.WorldGridIO.from_config(cfg)
    assert io.output.root == '/custom/grid'


# ----- `HarmonizationIO` tests
def test_harmonization_io():
    '''
    Given: Input grid and output harmonization paths.
    When: Instantiating `HarmonizationIO`.
    Then: Hold input and output path managers.
    '''
    grid_paths = paths.WorldGridPaths('/custom/grid')
    harm_paths = paths.HarmonizationPaths('/custom/harmonized')
    io = contracts.HarmonizationIO(grid=grid_paths, output=harm_paths)
    assert io.grid.root == '/custom/grid'
    assert io.output.root == '/custom/harmonized'


def test_harmonization_io_from_config():
    '''
    Given: A `RootConfig` with configured world grid and harmonization outputs.
    When: Instantiating `HarmonizationIO.from_config`.
    Then: Resolve upstream grid and harmonization output paths.
    '''
    cfg = configs.RootConfig()
    cfg.data.world_grid.output_dpath = '/custom/grid'
    cfg.data.harmonization.output_dpath = '/custom/harmonized'

    io = contracts.HarmonizationIO.from_config(cfg)
    assert io.grid.root == '/custom/grid'
    assert io.output.root == '/custom/harmonized'


def test_harmonization_io_missing_upstream_raises():
    '''
    Given: A `RootConfig` with empty upstream world grid output path.
    When: Calling `HarmonizationIO.from_config`.
    Then: Raise ValueError.
    '''
    cfg = configs.RootConfig()
    cfg.data.world_grid.output_dpath = ''

    with pytest.raises(ValueError, match='HarmonizationIO requires upstream'):
        contracts.HarmonizationIO.from_config(cfg)


# ----- `IngestionIO` tests
def test_ingestion_io():
    '''
    Given: Input harmonization and output ingestion paths.
    When: Instantiating `IngestionIO`.
    Then: Hold input and output path managers.
    '''
    harm_paths = paths.HarmonizationPaths('/custom/harmonized')
    ing_paths = paths.IngestionPaths('/custom/ingested')
    io = contracts.IngestionIO(harmonization=harm_paths, output=ing_paths)
    assert io.harmonization.root == '/custom/harmonized'
    assert io.output.root == '/custom/ingested'


def test_ingestion_io_from_config():
    '''
    Given: A `RootConfig` with configured harmonization and ingestion outputs.
    When: Instantiating `IngestionIO.from_config`.
    Then: Resolve upstream harmonization and ingestion output paths.
    '''
    cfg = configs.RootConfig()
    cfg.data.harmonization.output_dpath = '/custom/harmonized'
    cfg.data.ingestion.output_dpath = '/custom/ingested'

    io = contracts.IngestionIO.from_config(cfg)
    assert io.harmonization.root == '/custom/harmonized'
    assert io.output.root == '/custom/ingested'


def test_ingestion_io_missing_upstream_raises():
    '''
    Given: A `RootConfig` with empty upstream harmonization output path.
    When: Calling `IngestionIO.from_config`.
    Then: Raise ValueError.
    '''
    cfg = configs.RootConfig()
    cfg.data.harmonization.output_dpath = ''

    with pytest.raises(ValueError, match='IngestionIO requires upstream'):
        contracts.IngestionIO.from_config(cfg)


# ----- `PreparationIO` tests
def test_preparation_io():
    '''
    Given: Input ingestion and output preparation paths.
    When: Instantiating `PreparationIO`.
    Then: Hold input and output path managers.
    '''
    ing_paths = paths.IngestionPaths('/custom/ingested')
    prep_paths = paths.PreparationPaths('/custom/prepared')
    io = contracts.PreparationIO(ingestion=ing_paths, output=prep_paths)
    assert io.ingestion.root == '/custom/ingested'
    assert io.output.root == '/custom/prepared'


def test_preparation_io_from_config():
    '''
    Given: A `RootConfig` with configured ingestion and preparation outputs.
    When: Instantiating `PreparationIO.from_config`.
    Then: Resolve upstream ingestion and preparation output paths.
    '''
    cfg = configs.RootConfig()
    cfg.data.ingestion.output_dpath = '/custom/ingested'
    cfg.data.preparation.output_dpath = '/custom/prepared'

    io = contracts.PreparationIO.from_config(cfg)
    assert io.ingestion.root == '/custom/ingested'
    assert io.output.root == '/custom/prepared'


def test_preparation_io_missing_upstream_raises():
    '''
    Given: A `RootConfig` with empty upstream ingestion output path.
    When: Calling `PreparationIO.from_config`.
    Then: Raise ValueError.
    '''
    cfg = configs.RootConfig()
    cfg.data.ingestion.output_dpath = ''

    with pytest.raises(ValueError, match='PreparationIO requires upstream'):
        contracts.PreparationIO.from_config(cfg)


# ----- `TrainingIO` tests
def test_training_io():
    '''
    Given: Input preparation and output session paths.
    When: Instantiating `TrainingIO`.
    Then: Hold input and output path managers.
    '''
    prep_paths = paths.PreparationPaths('/custom/prepared')
    sess_paths = paths.SessionPaths('/custom/session')
    io = contracts.TrainingIO(preparation=prep_paths, output=sess_paths)
    assert io.preparation.root == '/custom/prepared'
    assert io.output.root == '/custom/session'


def test_training_io_from_config():
    '''
    Given: A `RootConfig` with configured preparation and session outputs.
    When: Instantiating `TrainingIO.from_config`.
    Then: Resolve upstream preparation and session output paths.
    '''
    cfg = configs.RootConfig()
    cfg.data.preparation.output_dpath = '/custom/prepared'
    cfg.session.output_dpath = '/custom/session'

    io = contracts.TrainingIO.from_config(cfg)
    assert io.preparation.root == '/custom/prepared'
    assert io.output.root == '/custom/session'


def test_training_io_missing_upstream_raises():
    '''
    Given: A `RootConfig` with empty upstream preparation output path.
    When: Calling `TrainingIO.from_config`.
    Then: Raise ValueError.
    '''
    cfg = configs.RootConfig()
    cfg.data.preparation.output_dpath = ''

    with pytest.raises(ValueError, match='TrainingIO requires upstream'):
        contracts.TrainingIO.from_config(cfg)


# ----- `EvaluationIO` tests
def test_evaluation_io():
    '''
    Given: Input preparation and output session paths.
    When: Instantiating `EvaluationIO`.
    Then: Hold input and output path managers.
    '''
    prep_paths = paths.PreparationPaths('/custom/prepared')
    sess_paths = paths.SessionPaths('/custom/session')
    io = contracts.EvaluationIO(preparation=prep_paths, output=sess_paths)
    assert io.preparation.root == '/custom/prepared'
    assert io.output.root == '/custom/session'


def test_evaluation_io_from_config():
    '''
    Given: A `RootConfig` with configured preparation and session outputs.
    When: Instantiating `EvaluationIO.from_config`.
    Then: Resolve upstream preparation and session output paths.
    '''
    cfg = configs.RootConfig()
    cfg.data.preparation.output_dpath = '/custom/prepared'
    cfg.session.output_dpath = '/custom/session'

    io = contracts.EvaluationIO.from_config(cfg)
    assert io.preparation.root == '/custom/prepared'
    assert io.output.root == '/custom/session'
