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

'''Unit tests for pipeline executor logic.'''

# pylint: disable=protected-access

# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.configs.schema.sections as secs
import landseg.execution.executor as executor


# ----- _validate_upstream_pipelines logic
def test_validate_upstream_pipelines_entrypoints():
    '''
    Given: Pipeline names that are standalone entrypoints.
    When: Validating upstream pipelines.
    Then: Pass silently without checking upstream reports.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='default'),
        data=secs.DataConfig()
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    # should not raise
    executor._validate_upstream_pipelines(config)
    config.pipeline.name = 'world-grid'
    executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_missing_world_grid(tmp_path):
    '''
    Given: Harmonization run when no world grid artifact exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError about missing world-grid.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='data-harmonize')
    )
    config.data.world_grid.output_dpath = str(tmp_path)
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "world-grid" has not been executed yet'
    ):
        executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_world_grid_success(tmp_path):
    '''
    Given: Harmonization run when world grid artifact exists.
    When: Validating upstream pipelines.
    Then: Pass silently.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='data-harmonize')
    )
    config.data.world_grid.output_dpath = str(tmp_path)
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    # create grid artifact JSON
    p = config.data.world_grid.params
    gid = (
        f'grid_row_{p.tile_size[0]}_{p.tile_stride[0]}'
        f'_col_{p.tile_size[1]}_{p.tile_stride[1]}'
    )
    grid_fp = tmp_path / f'{gid}.json'
    grid_fp.write_text('{}')

    # should not raise
    executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_world_grid_report_success(tmp_path):
    '''
    Given: A data-harmonize pipeline run when grid_report.json exists.
    When: Validating upstream pipelines.
    Then: Pass silently.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='data-harmonize')
    )
    config.data.world_grid.output_dpath = str(tmp_path)
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    report_fp = tmp_path / 'grid_report.json'
    report_fp.write_text('{"grid_fpath": "some_path", "grid_id": "test"}')

    # should not raise
    executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_missing_etl(tmp_path):
    '''
    Given: A data-ingest pipeline run when no harmonization report exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError about missing data-harmonize.
    '''
    config = configs.RootConfig(pipeline=secs.PipelineConfig(name='data-ingest'))

    config.data.harmonization.output_dpath = tmp_path
    config.data.ingestion.output_dpath = tmp_path
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "data-harmonize" has not been executed yet'
    ):
        executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_missing_ingest(tmp_path):
    '''
    Given: A pipeline that requires data-ingest to be complete, but no
        report exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError.
    '''
    harmonize_paths = artifacts.HarmonizationPaths(str(tmp_path))
    artifacts.Controller(harmonize_paths.report).persist({'status': 'SUCCESS'})

    config = configs.RootConfig(pipeline=secs.PipelineConfig(name='data-prepare'))

    config.data.harmonization.output_dpath = tmp_path
    config.data.ingestion.output_dpath = tmp_path
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "data-ingest" has not been executed yet'
    ):
        executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_failed_ingest(tmp_path):
    '''
    Given: A pipeline that requires data-ingest, and a failed ingest
        report exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError about the status.
    '''
    etl_paths = artifacts.HarmonizationPaths(str(tmp_path))
    artifacts.Controller(etl_paths.report).persist({'status': 'SUCCESS'})
    ingest_paths = artifacts.IngestionPaths(str(tmp_path))
    ctrl = artifacts.Controller(ingest_paths.report)
    ctrl.persist({'status': 'FAILED'})

    config = configs.RootConfig(pipeline=secs.PipelineConfig(name='data-prepare'))

    config.data.harmonization.output_dpath = tmp_path
    config.data.ingestion.output_dpath = tmp_path
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "data-ingest" status is "FAILED"'
    ):
        executor._validate_upstream_pipelines(config)


def test_validate_upstream_pipelines_success(tmp_path):
    '''
    Given: Successful etl, ingest and prepare reports exist.
    When: Validating upstream pipelines for a downstream pipeline.
    Then: Pass silently.
    '''
    harmonization_paths = artifacts.HarmonizationPaths(str(tmp_path))
    ingestion_paths = artifacts.IngestionPaths(str(tmp_path))
    preparation_paths = artifacts.PreparationPaths(str(tmp_path))

    artifacts.Controller(harmonization_paths.report).persist({'status': 'SUCCESS'})
    artifacts.Controller(ingestion_paths.report).persist({'status': 'SUCCESS'})
    artifacts.Controller(preparation_paths.report).persist({'status': 'SUCCESS'})

    config = configs.RootConfig(pipeline=secs.PipelineConfig(name='model-train'))

    config.data.harmonization.output_dpath = tmp_path
    config.data.ingestion.output_dpath = tmp_path
    config.data.preparation.output_dpath = tmp_path
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    # should not raise
    executor._validate_upstream_pipelines(config)


# ----- execute_pipeline logic
def test_execute_pipeline_success(mocker):
    '''
    Given: A valid root configuration and all upstream validation checks
        passing.
    When: Calling execute_pipeline.
    Then: Retrieve the correct pipeline command, execute it with the
        configuration, and return its result.
    '''
    config = configs.RootConfig(pipeline=secs.PipelineConfig(name='default'))

    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1

    mocker.patch('landseg.execution.executor._validate_upstream_pipelines')

    mock_command = mocker.Mock(return_value='pipeline_result')
    mocker.patch('landseg.execution.pipelines.get', return_value=mock_command)

    result = executor.execute_pipeline(config)

    assert result == 'pipeline_result'
    mock_command.assert_called_once_with(config)
