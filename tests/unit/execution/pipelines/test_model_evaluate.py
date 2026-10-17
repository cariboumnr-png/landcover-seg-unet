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

'''Unit tests for model evaluate pipeline (model_evaluate.py).'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines.model_evaluate as eval_pipeline


# ----- `ModelEvaluation` configuration validation tests
def test_evaluate_invalid_split_raises_value_error(tmp_path):
    '''
    Given: A RootConfig with an invalid evaluation split.
    When: `config.session.evaluation.validate()` is called.
    Then: Raise a ConfigValidationError.
    '''
    chk_file = tmp_path / 'chk.pt'
    chk_file.write_text('dummy', encoding='utf-8')

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path
    schema.session.evaluation.checkpoint = chk_file
    schema.session.evaluation.split = 'invalid'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    with pytest.raises(configs.schema.base.ConfigValidationError):
        config.session.evaluation.validate()


def test_evaluate_validate_missing_checkpoint(tmp_path):
    '''
    Given: Evaluation configuration with non-existent checkpoint path.
    When: Calling `config.session.evaluation.validate()` on RootConfig.
    Then: Raise a ConfigValidationError.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path
    schema.session.evaluation.checkpoint = tmp_path / 'missing.pt'
    schema.session.evaluation.split = 'val'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    with pytest.raises(configs.schema.base.ConfigValidationError):
        config.session.evaluation.validate()


def test_evaluate_validate_missing_prep_report(tmp_path):
    '''
    Given: Evaluation configuration without upstream preparation report.
    When: Calling `validate` on `ModelEvaluation`.
    Then: Raise a RuntimeError indicating missing preparation report.
    '''
    chk_file = tmp_path / 'chk.pt'
    chk_file.write_text('dummy', encoding='utf-8')

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path
    schema.data.preparation.output_dpath = tmp_path / 'prep'
    schema.session.evaluation.checkpoint = chk_file
    schema.session.evaluation.split = 'val'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = eval_pipeline.ModelEvaluation(config)
    with pytest.raises(
        RuntimeError,
        match='Upstream pipeline "data-prepare"'
    ):
        pipeline._validate()


# ----- `ModelEvaluation` execution and lifecycle tests
def test_evaluate_pipeline_success(tmp_path, mocker):
    '''
    Given: A valid RootConfig and mock session runner.
    When: `ModelEvaluation.run` is called.
    Then: Execute evaluation and persist evaluation metrics artifact.
    '''
    exp_root = tmp_path / 'exp'
    chk_file = tmp_path / 'chk.pt'
    chk_file.write_text('dummy', encoding='utf-8')

    prep_root = tmp_path / 'prep'
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = prep_root
    schema.session.evaluation.checkpoint = chk_file
    schema.session.evaluation.split = 'val'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    mock_metric = mocker.Mock(as_dict={'iou': 0.85})
    mock_runner = mocker.Mock()
    mock_runner.run_epoch.return_value = mocker.Mock(
        validation=mocker.Mock(head_metrics={'head_1': mock_metric}),
        target_metrics=0.85,
    )

    mocker.patch.object(
        eval_pipeline.ModelEvaluation,
        'build_session_runner',
        return_value=mock_runner,
    )

    target_metric = eval_pipeline.ModelEvaluation(config).run()

    assert target_metric == 0.85
    eval_fpath = exp_root / 'results' / 'run_0001' / 'evaluation.json'
    assert eval_fpath.exists()
    assert artifacts.Controller[dict](eval_fpath).fetch() == {
        'head_1': {'iou': 0.85}
    }

    report_fpath = exp_root / 'results' / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'SUCCESS'
    assert report['results'] == {'final': 0.85}


def test_evaluate_pipeline_failure(tmp_path, mocker):
    '''
    Given: Evaluation pipeline where the runner fails.
    When: `ModelEvaluation.run` is called.
    Then: Re-raise exception and mark report status as FAILED.
    '''
    exp_root = tmp_path / 'exp'
    chk_file = tmp_path / 'chk.pt'
    chk_file.write_text('dummy', encoding='utf-8')

    prep_root = tmp_path / 'prep'
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = prep_root
    schema.session.evaluation.checkpoint = chk_file
    schema.session.evaluation.split = 'val'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    mock_runner = mocker.Mock()
    mock_runner.run_epoch.side_effect = RuntimeError('evaluation error')
    mocker.patch.object(
        eval_pipeline.ModelEvaluation,
        'build_session_runner',
        return_value=mock_runner,
    )

    pipeline = eval_pipeline.ModelEvaluation(config)
    with pytest.raises(RuntimeError, match='evaluation error'):
        pipeline.run()

    report_fpath = exp_root / 'results' / 'run_0001' / 'report.json'
    assert report_fpath.exists()
    report = artifacts.Controller[dict](report_fpath).fetch()
    assert report
    assert report['status'] == 'FAILED'


def test_evaluate_build_context_empty_split_raises(tmp_path, mocker, dataspecs):
    '''
    Given: DataSpecs whose evaluation split is empty.
    When: `_build_context` is executed by ModelEvaluation.
    Then: Raise RuntimeError indicating split has no blocks.
    '''
    prep_root = tmp_path / 'prep'
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    chk_file = tmp_path / 'chk.pt'
    chk_file.write_text('dummy', encoding='utf-8')

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = tmp_path / 'exp'
    schema.data.preparation.output_dpath = prep_root
    schema.session.evaluation.checkpoint = chk_file
    schema.session.evaluation.split = 'test'

    # simulate empty test split
    dataspecs.splits.test.clear()
    mocker.patch.object(
        eval_pipeline.geopipe,
        'build_dataspec',
        return_value=dataspecs,
    )

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = eval_pipeline.ModelEvaluation(config)
    with pytest.raises(RuntimeError, match='has no blocks in prepared dataset'):
        pipeline._build_context()
