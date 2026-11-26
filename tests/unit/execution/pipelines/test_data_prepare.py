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

'''Unit tests for the data preparation execution pipeline.'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.geopipe as geopipe


# ----- `DataPreparation` validation tests
def test_data_prepare_validate_missing_catalog_or_schema(tmp_path):
    '''
    Given: Preparation configuration without upstream catalog/schema.
    When: Calling `validate` on `DataPreparation`.
    Then: Raise a RuntimeError indicating missing data-ingest artifacts.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested')
    cfg_schema.data.preparation.output_dpath = str(tmp_path / 'prepared')

    config = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg_schema))
    pipeline = pipelines.DataPreparation(config)
    with pytest.raises(RuntimeError, match='Upstream pipeline "data-ingest"'):
        pipeline.run()


# ----- `DataPreparation` execution and lifecycle tests
def test_data_prepare_pipeline_success(tmp_path, mocker):
    '''
    Given: Upstream ingestion artifacts and mocked preparation stage.
    When: Calling `run` on `DataPreparation`.
    Then: Initialize folders/config and delegate execution to `geopipe`.
    '''
    ingest_dpath = tmp_path / 'ingested'
    catalog_fpath = ingest_dpath / 'data_blocks' / 'catalog.json'
    schema_fpath = ingest_dpath / 'data_blocks' / 'schema.json'
    runs_fpath = ingest_dpath / 'ingestion_runs.json'

    artifacts.Controller[dict](catalog_fpath).persist({'test_block': {}})
    artifacts.Controller[dict](schema_fpath).persist({})
    artifacts.Controller[list](runs_fpath).persist(
        [{'status': 'SUCCESS', 'run_id': 'run_0001'}]
    )

    prep_dpath = tmp_path / 'prepared'
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.ingestion.output_dpath = str(ingest_dpath)
    cfg_schema.data.preparation.output_dpath = str(prep_dpath)

    config = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg_schema))

    mock_stage = mocker.patch.object(geopipe, 'run_data_preparation_stage')
    mocker.patch('landseg.execution.preflight.assert_target_ready')

    pipeline = pipelines.DataPreparation(config)
    pipeline.run()

    mock_stage.assert_called_once_with(
        config.data.preparation,
        pipeline.upstream_paths,
        pipeline.pipeline_paths,
    )
    assert (prep_dpath / 'config.json').exists()


def test_data_prepare_pipeline_failure(tmp_path, mocker):
    '''
    Given: Upstream ingestion artifacts and stage runner raising error.
    When: Calling `run` on `DataPreparation`.
    Then: Propagate error to caller cleanly.
    '''
    ingest_dpath = tmp_path / 'ingested'
    catalog_fpath = ingest_dpath / 'data_blocks' / 'catalog.json'
    schema_fpath = ingest_dpath / 'data_blocks' / 'schema.json'
    runs_fpath = ingest_dpath / 'ingestion_runs.json'

    artifacts.Controller[dict](catalog_fpath).persist({'test_block': {}})
    artifacts.Controller[dict](schema_fpath).persist({})
    artifacts.Controller[list](runs_fpath).persist(
        [{'status': 'SUCCESS', 'run_id': 'run_0001'}]
    )

    prep_dpath = tmp_path / 'prepared'
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.ingestion.output_dpath = str(ingest_dpath)
    cfg_schema.data.preparation.output_dpath = str(prep_dpath)

    config = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg_schema))

    mocker.patch.object(
        geopipe,
        'run_data_preparation_stage',
        side_effect=RuntimeError('partitioning failure'),
    )
    mocker.patch('landseg.execution.preflight.assert_target_ready')

    pipeline = pipelines.DataPreparation(config)
    with pytest.raises(RuntimeError, match='partitioning failure'):
        pipeline.run()
