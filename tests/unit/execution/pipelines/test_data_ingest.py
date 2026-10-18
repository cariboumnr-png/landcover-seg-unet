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

'''Unit tests for the data ingestion execution pipeline.'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.geopipe.ingest as ingest


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

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )
    pipeline = pipelines.DataIngestion(config)
    with pytest.raises(
        RuntimeError,
        match='Upstream pipeline "data-harmonize"'
    ):
        pipeline._validate()


# ----- `DataIngestion` execution and lifecycle tests
def test_data_ingest_pipeline_success(tmp_path, mocker):
    '''
    Given: Upstream harmonization runs and mocked ingest runner.
    When: Calling `run` on `DataIngestion`.
    Then: Execute ingestion, update manifest, and write SUCCESS report.
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

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    mock_context = mocker.Mock(
        harmonization_run_id='run_0001',
        run_uid='hash_123',
    )
    mocker.patch.object(
        pipelines.DataIngestion,
        '_build_context',
        return_value=mock_context,
    )
    mock_run = mocker.patch.object(ingest, 'run_data_ingestion')

    pipeline = pipelines.DataIngestion(config)
    pipeline.run()

    mock_run.assert_called_once()
    report_fpath = ingest_dpath / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'SUCCESS'

    runs_fpath = ingest_dpath / 'ingestion_runs.json'
    assert runs_fpath.exists()
    runs = artifacts.Controller[dict](runs_fpath).fetch()
    assert runs
    uids = list(runs.keys())
    assert runs[uids[0]]['status'] == 'SUCCESS'


def test_data_ingest_pipeline_failure(tmp_path, mocker):
    '''
    Given: Upstream harmonization runs and runner that raises an error.
    When: Calling `run` on `DataIngestion`.
    Then: Re-raise error and mark report status as FAILED.
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

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    mock_context = mocker.Mock(
        harmonization_run_id='run_0001',
        run_uid='hash_123',
    )
    mocker.patch.object(
        pipelines.DataIngestion,
        '_build_context',
        return_value=mock_context,
    )
    mocker.patch.object(
        ingest,
        'run_data_ingestion',
        side_effect=RuntimeError('Ingestion failed unexpectedly')
    )

    pipeline = pipelines.DataIngestion(config)
    with pytest.raises(RuntimeError, match='Ingestion failed unexpectedly'):
        pipeline.run()

    report_fpath = ingest_dpath / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'FAILED'
