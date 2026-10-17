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
import landseg.geopipe.prepare as prepare


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

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )
    pipeline = pipelines.DataPreparation(config)
    with pytest.raises(
        RuntimeError,
        match='Upstream pipeline "data-ingest"'
    ):
        pipeline.run()


# ----- `DataPreparation` execution and lifecycle tests
def test_data_prepare_pipeline_success(tmp_path, mocker):
    '''
    Given: Upstream ingestion artifacts and mocked prepare runner.
    When: Calling `run` on `DataPreparation`.
    Then: Execute preparation and persist report with status SUCCESS.
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

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    mock_context = mocker.Mock()
    mocker.patch.object(
        pipelines.DataPreparation,
        '_build_context',
        return_value=mock_context,
    )
    mock_run = mocker.patch.object(prepare, 'run_data_preparation')

    pipeline = pipelines.DataPreparation(config)
    pipeline.run()

    mock_run.assert_called_once()
    report_fpath = prep_dpath / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'SUCCESS'


def test_data_prepare_pipeline_failure(tmp_path, mocker):
    '''
    Given: Upstream ingestion artifacts and runner that raises an error.
    When: Calling `run` on `DataPreparation`.
    Then: Re-raise error and mark report status as FAILED.
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

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    mock_context = mocker.Mock()
    mocker.patch.object(
        pipelines.DataPreparation,
        '_build_context',
        return_value=mock_context,
    )
    mocker.patch.object(
        prepare,
        'run_data_preparation',
        side_effect=RuntimeError('partitioning failure'),
    )

    pipeline = pipelines.DataPreparation(config)
    with pytest.raises(RuntimeError, match='partitioning failure'):
        pipeline.run()

    report_fpath = prep_dpath / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'FAILED'
