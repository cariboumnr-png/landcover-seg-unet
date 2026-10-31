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
Unit tests for the model training execution pipeline.

These tests verify pipeline lifecycle, configuration validation, and the
session runner interaction contract. Complete end-to-end training with
real tensors, losses, and checkpoints is tested in integration tests.
'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines.model_train as train_pipeline
import landseg.geopipe as geopipe


# ----- `ModelTraining` validation tests
def test_train_validate_missing_preparation_report(tmp_path):
    '''
    Given: Training configuration without upstream preparation report.
    When: Calling `validate` on `ModelTraining`.
    Then: Raise a RuntimeError indicating missing preparation report.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path / 'exp'
    schema.data.preparation.output_dpath = tmp_path / 'prep'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = train_pipeline.ModelTraining(config)
    with pytest.raises(RuntimeError, match='Upstream pipeline "data-prepare"'):
        pipeline.run()


def test_train_validate_failed_preparation_report(tmp_path):
    '''
    Given: Training configuration with a failed preparation report.
    When: Calling `validate` on `ModelTraining`.
    Then: Raise a RuntimeError indicating failed preparation status.
    '''
    prep_root = tmp_path / 'prep'
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'FAILED'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path / 'exp'
    schema.data.preparation.output_dpath = prep_root

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = train_pipeline.ModelTraining(config)
    with pytest.raises(
        RuntimeError,
        match='Upstream pipeline "data-prepare" status is "FAILED"'
    ):
        pipeline.run()


# ----- `ModelTraining` execution and lifecycle tests


def test_train_pipeline_success(tmp_path, mocker):
    '''
    Given: A valid RootConfig and mock training session runner.
    When: Calling `run` on `ModelTraining`.
    Then: Execute training session and persist successful report.
    '''
    exp_root = tmp_path / 'exp'
    prep_root = tmp_path / 'prep'
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = prep_root

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    mock_runner = mocker.Mock()
    mock_runner.execute.return_value = 0.92

    mocker.patch.object(
        train_pipeline.ModelTraining,
        'build_session_runner',
        return_value=mock_runner,
    )

    mocker.patch('landseg.execution.preflight.assert_target_ready')
    pipeline = train_pipeline.ModelTraining(config)
    pipeline.run()

    report_fpath = exp_root / 'results' / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'SUCCESS'
    assert report['results']['best_value'] == 0.92
    assert 'durations' in report['results']
    assert 'system' in report['results']


def test_train_pipeline_failure(tmp_path, mocker):
    '''
    Given: A training session runner that raises a RuntimeError.
    When: Calling `run` on `ModelTraining`.
    Then: Re-raise error and mark report status as FAILED.
    '''
    exp_root = tmp_path / 'exp'
    prep_root = tmp_path / 'prep'
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = prep_root

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    mock_runner = mocker.Mock()
    mock_runner.execute.side_effect = RuntimeError('CUDA out of memory')

    mocker.patch.object(
        train_pipeline.ModelTraining,
        'build_session_runner',
        return_value=mock_runner,
    )

    mocker.patch('landseg.execution.preflight.assert_target_ready')
    pipeline = train_pipeline.ModelTraining(config)
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        pipeline.run()

    report_fpath = exp_root / 'results' / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'FAILED'


def test_train_summarize_results(tmp_path):
    '''
    Given: A ModelTraining instance with populated timer entries.
    When: Calling `_summarize_results` with final best metric value.
    Then: Return structured summary with durations and system metrics.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path / 'exp'
    schema.data.preparation.output_dpath = tmp_path / 'prep'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = train_pipeline.ModelTraining(config)
    pipeline.timer['data'] = 1.5
    pipeline.timer['model'] = 0.5
    pipeline.timer['exec'] = 10.0

    summary = pipeline._summarize_results(0.88)
    assert summary['best_value'] == 0.88
    assert summary['duration_sec'] == pytest.approx(12.0)
    assert summary['durations']['data_specs_setup_sec'] == 1.5
    assert summary['durations']['model_assembly_sec'] == 0.5
    assert summary['durations']['execution_sec'] == 10.0
    assert 'peak_cpu_memory_mb' in summary['system']
    assert 'peak_gpu_memory_mb' in summary['system']


def test_train_build_context(tmp_path, mocker):
    '''
    Given: Configured domain IDs and vector domain names.
    When: Calling `_build_context` on `ModelTraining`.
    Then: Delegate to `geopipe.build_dataspec` with domain parameters.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path / 'exp'
    schema.data.preparation.output_dpath = tmp_path / 'prep'
    schema.data.specification.domain_ids_name = 'discrete_domain'
    schema.data.specification.domain_vec_name = 'continuous_domain'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    mock_build_dataspec = mocker.patch.object(
        geopipe,
        'build_dataspec',
        return_value=mocker.Mock(),
    )

    pipeline = train_pipeline.ModelTraining(config)
    pipeline._build_context()

    mock_build_dataspec.assert_called_once_with(
        pipeline.artifact_paths,
        mode='default',
        ids_domain_name='discrete_domain',
        vec_domain_name='continuous_domain',
    )
