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

'''Unit tests for the data ingestion execution pipeline.'''

# pylint: disable=protected-access

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


# ----- `DataIngestion` validation tests
def test_data_ingest_validate_missing_harmonization(tmp_path):
    '''
    Given: Ingestion configuration without upstream harmonization runs.
    When: Calling `validate` on `DataIngestion`.
    Then: Raise a RuntimeError indicating missing data-harmonize runs.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')
    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested')

    config = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg_schema))
    pipeline = pipelines.DataIngestion(config)
    with pytest.raises(
        RuntimeError,
        match='Upstream pipeline "data-harmonize"'
    ):
        pipeline._validate()


# ----- `DataIngestion` execution and lifecycle tests
def test_data_ingest_pipeline_success(tmp_path, mocker):
    '''
    Given: Upstream harmonization runs and mocked ingestion stage.
    When: Calling `run` on `DataIngestion`.
    Then: Initialize folders/config and forward record and settings to
        `geopipe`.
    '''
    harm_dpath = tmp_path / 'harmonized'
    runs_fpath = harm_dpath / 'harmonization_runs.json'
    artifacts.Controller[dict](runs_fpath).persist({
        'hash_123': {'status': 'SUCCESS', 'run_id': 'run_0001'}
    })

    ingest_dpath = tmp_path / 'ingested_data'
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.harmonization.output_dpath = str(harm_dpath)
    cfg_schema.data.ingestion.output_dpath = str(ingest_dpath)
    cfg_schema.data.ingestion.rebuild = False

    config = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg_schema))

    mock_stage = mocker.patch.object(geopipe, 'run_data_ingestion_stage')
    mocker.patch('landseg.execution.preflight.assert_target_ready')

    record = {'run_uid': 'hash_123', 'run_id': 'run_0001'}
    pipeline = pipelines.DataIngestion(config)
    pipeline.run(record)

    mock_stage.assert_called_once_with(
        config.data.ingestion,
        pipeline.upstream_paths,
        pipeline.pipeline_paths,
        harmonization_run=config.data.ingestion.harmonization_run,
        rebuild=False,
        harmonization_record=record,
    )
    assert (ingest_dpath / 'run_0001' / 'config.json').exists()


def test_data_ingest_pipeline_failure(tmp_path, mocker):
    '''
    Given: Upstream harmonization runs and stage runner raising error.
    When: Calling `run` on `DataIngestion`.
    Then: Propagate error to caller cleanly.
    '''
    harm_dpath = tmp_path / 'harmonized'
    runs_fpath = harm_dpath / 'harmonization_runs.json'
    artifacts.Controller[dict](runs_fpath).persist({
        'hash_123': {'status': 'SUCCESS', 'run_id': 'run_0001'}
    })

    ingest_dpath = tmp_path / 'ingested_data'
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.harmonization.output_dpath = str(harm_dpath)
    cfg_schema.data.ingestion.output_dpath = str(ingest_dpath)

    config = typing.cast(configs.RootConfig, omegaconf.OmegaConf.to_object(cfg_schema))

    mocker.patch.object(
        geopipe,
        'run_data_ingestion_stage',
        side_effect=RuntimeError('Ingestion failed unexpectedly')
    )
    mocker.patch('landseg.execution.preflight.assert_target_ready')

    pipeline = pipelines.DataIngestion(config)
    with pytest.raises(RuntimeError, match='Ingestion failed unexpectedly'):
        pipeline.run()
