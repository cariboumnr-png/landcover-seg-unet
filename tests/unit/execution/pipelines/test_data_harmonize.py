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

'''Unit tests for the data harmonization execution pipeline.'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.geopipe.harmonize as harmonize


# ----- `DataHarmonization` validation tests
def test_data_harmonize_validate_missing_world_grid(tmp_path):
    '''
    Given: Harmonization configuration without upstream world grid report.
    When: Calling `validate` on `DataHarmonization`.
    Then: Raise a RuntimeError indicating missing world-grid report.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.world_grid.output_dpath = str(tmp_path / 'world_grids')
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )
    pipeline = pipelines.DataHarmonization(config)
    with pytest.raises(RuntimeError, match='Upstream pipeline "world-grid"'):
        pipeline._validate()


# ----- `DataHarmonization` execution and lifecycle tests
def test_data_harmonize_pipeline_success(tmp_path, mocker):
    '''
    Given: Upstream world-grid report and mocked harmonization runner.
    When: Calling `run` on `DataHarmonization`.
    Then: Execute harmonization, update manifest, and write SUCCESS report.
    '''
    grid_dpath = tmp_path / 'world_grids'
    grid_report = grid_dpath / 'report.json'
    artifacts.Controller[dict](grid_report).persist({'status': 'SUCCESS'})

    manifest_fpath = tmp_path / 'dataset_manifest.json'
    manifest_fpath.write_text('{}', encoding='utf-8')

    harm_dpath = tmp_path / 'harmonized'
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.world_grid.output_dpath = str(grid_dpath)
    cfg_schema.data.harmonization.output_dpath = str(harm_dpath)
    cfg_schema.data.harmonization.dataset_manifest = str(manifest_fpath)

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    mock_context = mocker.Mock(is_collision=False)
    mocker.patch.object(
        harmonize,
        'build_harmonization_context',
        return_value=mock_context,
    )
    mock_run = mocker.patch.object(harmonize, 'run_data_harmonization')
    mocker.patch('landseg.execution.preflight.assert_target_ready')

    pipeline = pipelines.DataHarmonization(config)
    pipeline.run()

    mock_run.assert_called_once()
    report_fpath = harm_dpath / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'SUCCESS'

    runs_fpath = harm_dpath / 'harmonization_runs.json'
    assert runs_fpath.exists()
    runs = artifacts.Controller[dict](runs_fpath).fetch()
    assert runs


def test_data_harmonize_pipeline_failure(tmp_path, mocker):
    '''
    Given: Upstream world-grid report and runner that raises an error.
    When: Calling `run` on `DataHarmonization`.
    Then: Re-raise error and mark report status as FAILED.
    '''
    grid_dpath = tmp_path / 'world_grids'
    grid_report = grid_dpath / 'report.json'
    artifacts.Controller[dict](grid_report).persist({'status': 'SUCCESS'})

    manifest_fpath = tmp_path / 'dataset_manifest.json'
    manifest_fpath.write_text('{}', encoding='utf-8')

    harm_dpath = tmp_path / 'harmonized'
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path / 'exp')
    cfg_schema.data.world_grid.output_dpath = str(grid_dpath)
    cfg_schema.data.harmonization.output_dpath = str(harm_dpath)
    cfg_schema.data.harmonization.dataset_manifest = str(manifest_fpath)

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    mock_context = mocker.Mock(is_collision=False)
    mocker.patch.object(
        harmonize,
        'build_harmonization_context',
        return_value=mock_context,
    )
    mocker.patch.object(
        harmonize,
        'run_data_harmonization',
        side_effect=RuntimeError('Harmonization failed unexpectedly')
    )
    mocker.patch('landseg.execution.preflight.assert_target_ready')

    pipeline = pipelines.DataHarmonization(config)
    with pytest.raises(RuntimeError, match='Harmonization failed unexpectedly'):
        pipeline.run()

    report_fpath = harm_dpath / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'FAILED'
