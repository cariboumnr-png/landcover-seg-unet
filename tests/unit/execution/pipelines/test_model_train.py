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

'''Unit tests for model training execution pipeline.'''

# pylint: disable=protected-access

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
import landseg.session as session


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
        pipeline.artifact_paths.data_preparation,
        mode='default',
        ids_domain_name='discrete_domain',
        vec_domain_name='continuous_domain',
        knowledge_root=pipeline.artifact_paths.knowledge_root,
    )


def test_train_generate_post_run_visuals(tmp_path, mocker):
    '''
    Given: `ModelTraining` pipeline with finished run, `step_results`,
        and test split.
    When: Calling `_generate_post_run_visuals`.
    Then: Delegate curve plotting, preview inference, and preview
        rendering.
    '''
    exp_root = tmp_path / 'exp'
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = tmp_path / 'prep'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = train_pipeline.ModelTraining(config)
    pipeline.pipeline_paths = artifacts.SessionPaths.allocate_run(
        pipeline.pipeline_paths.root
    ).init_pipeline_folders()
    pipeline.logger = mocker.Mock()

    # Persist step_results.json via Controller so hash integrity validates
    step_data = [{'epoch_in_phase': 1, 'training': {'total_objective': 1.0}}]
    artifacts.Controller[list](pipeline.pipeline_paths.step_results).persist(
        step_data
    )

    # Create best checkpoint and populate CheckpointCatalog
    best_ckpt = pipeline.pipeline_paths.best_checkpoint('base_epoch')
    last_ckpt = pipeline.pipeline_paths.last_checkpoint('base_epoch')
    best_ckpt.write_bytes(b'ckpt')
    last_ckpt.write_bytes(b'ckpt')
    catalog = session.CheckpointCatalog(run=pipeline.pipeline_paths)
    catalog.record_save(
        'base',
        last_fpath=last_ckpt,
        epoch=1,
        best_fpath=best_ckpt,
        metric=0.80,
    )
    catalog.mark_phase_completed('base')
    catalog.persist()

    # Mock dataspecs with test split
    mock_specs = mocker.Mock()
    mock_specs.splits.test = {'blk1': 'path1'}
    pipeline.context = mock_specs

    mock_model = mocker.Mock()
    mock_infer_results = mocker.Mock()
    mock_infer_results.infer_preds = {'head1': mocker.Mock()}
    mocker.patch.object(pipeline, '_build_model', return_value=mock_model)
    mock_run_infer = mocker.patch(
        'landseg.session.run_preview_inference',
        return_value=mock_infer_results,
    )
    mock_gen_curves = mocker.patch('landseg.visual.generate_training_curves')
    mock_gen_previews = mocker.patch(
        'landseg.visual.generate_prediction_previews',
        return_value={'head1': pipeline.pipeline_paths.preview('head1')},
    )

    pipeline._generate_post_run_visuals()

    mock_gen_curves.assert_called_once_with(
        step_data,
        pipeline.pipeline_paths.training_curves,
    )
    mock_run_infer.assert_called_once_with(
        checkpoint_fpath=best_ckpt,
        dataspecs=mock_specs,
        model=mock_model,
        config=pipeline.config.session,
        device=pipeline.config.session.device,
    )
    mock_gen_previews.assert_called_once_with(
        mock_infer_results,
        mock_specs,
        {'head1': pipeline.pipeline_paths.preview('head1')},
    )


def test_train_generate_post_run_visuals_skips_when_infer_none(tmp_path, mocker):
    '''
    Given: `ModelTraining` where preview inference returns `None`.
    When: Calling `_generate_post_run_visuals`.
    Then: Skip calling `visual.generate_prediction_previews`.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path / 'exp'
    schema.data.preparation.output_dpath = tmp_path / 'prep'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = train_pipeline.ModelTraining(config)
    pipeline.pipeline_paths = artifacts.SessionPaths.allocate_run(
        pipeline.pipeline_paths.root
    )
    pipeline.logger = mocker.Mock()

    pipeline.pipeline_paths.checkpoints.mkdir(parents=True, exist_ok=True)
    best_ckpt = pipeline.pipeline_paths.checkpoints / 'checkpoint_epoch_0005_best.pt'
    best_ckpt.write_bytes(b'ckpt')

    mock_specs = mocker.Mock()
    mock_specs.splits.test = {'blk1': 'path1'}
    pipeline.context = mock_specs

    mocker.patch.object(pipeline, '_build_model', return_value=mocker.Mock())
    mocker.patch('landseg.session.run_preview_inference', return_value=None)
    mock_gen_previews = mocker.patch('landseg.visual.generate_prediction_previews')

    pipeline._generate_post_run_visuals()

    mock_gen_previews.assert_not_called()
