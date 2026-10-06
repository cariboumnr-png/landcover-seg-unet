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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Unit tests for model evaluate pipeline (model_evaluate.py).
'''

# standard imports
import dataclasses
import os
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.execution.pipelines.model_evaluate as eval_pipeline
import landseg.models as models
import landseg.session as session


# ----- `ModelEvaluation` tests
def test_evaluate_invalid_split_raises_value_error(tmp_path):
    '''
    Given: A RootConfig with an invalid evaluation split.
    When: `config.command.validate()` is called.
    Then: Raise a ValueError.
    '''
    chk_file = str(tmp_path / 'chk.pt')
    with open(chk_file, 'w', encoding='utf-8') as f:
        f.write('dummy')

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    schema.command.model_evaluate.checkpoint = chk_file
    schema.command.model_evaluate.split = 'invalid'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    with pytest.raises(ValueError, match='Invalid split'):
        config.command.validate()


def test_evaluate_pipeline_success(tmp_path, dataspecs, monkeypatch):
    '''
    Given: A valid RootConfig and mock dependencies.
    When: `ModelEvaluation.run` is called.
    Then: Execute evaluation and persist results.
    '''
    exp_root = str(tmp_path / 'exp')
    chk_file = str(tmp_path / 'chk.pt')
    with open(chk_file, 'w', encoding='utf-8') as f:
        f.write('dummy')

    prep_root = str(tmp_path / 'prep')
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = prep_root
    schema.command.model_evaluate.checkpoint = chk_file
    schema.command.model_evaluate.split = 'val'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    @dataclasses.dataclass
    class MockHeadMetrics:
        as_dict: dict = dataclasses.field(
            default_factory=lambda: {'iou': 0.85}
        )

    @dataclasses.dataclass
    class MockValidation:
        head_metrics: dict = dataclasses.field(
            default_factory=lambda: {'head_1': MockHeadMetrics()}
        )

    @dataclasses.dataclass
    class MockEvalResult:
        validation: MockValidation = dataclasses.field(
            default_factory=MockValidation
        )
        target_metrics: float = 0.85

    class MockRunner:
        def run_epoch(self, _epoch: int):
            return MockEvalResult()

    class DummyModel:
        def parameters(self):
            return []

    mock_model = typing.cast(core.MultiheadModelLike, DummyModel())

    monkeypatch.setattr(
        eval_pipeline.geopipe, 'build_dataspec', lambda *a, **kw: dataspecs
    )
    monkeypatch.setattr(
        models, 'build_multihead_unet', lambda *a, **kw: mock_model
    )
    monkeypatch.setattr(
        session,
        'build_session_runner',
        lambda *a, **kw: MockRunner(),
    )

    target_metric = eval_pipeline.ModelEvaluation(config).run()

    assert target_metric == 0.85
    assert os.path.exists(f'{exp_root}/results/run_0001/evaluation.json')


def test_evaluate_validate_missing_checkpoint(tmp_path):
    '''
    Given: Evaluation configuration with non-existent checkpoint path.
    When: Calling `config.command.validate()` on RootConfig.
    Then: Raise a FileNotFoundError.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    schema.command.model_evaluate.checkpoint = str(tmp_path / 'missing.pt')
    schema.command.model_evaluate.split = 'val'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    with pytest.raises(FileNotFoundError, match='Checkpoint not found'):
        config.command.validate()


def test_evaluate_validate_missing_prep_report(tmp_path):
    '''
    Given: Evaluation configuration without upstream preparation report.
    When: Calling `validate` on `ModelEvaluation`.
    Then: Raise a RuntimeError indicating missing preparation report.
    '''
    chk_file = str(tmp_path / 'chk.pt')
    with open(chk_file, 'w', encoding='utf-8') as f:
        f.write('dummy')

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    schema.data.preparation.output_dpath = str(tmp_path / 'prep')
    schema.command.model_evaluate.checkpoint = chk_file
    schema.command.model_evaluate.split = 'val'

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
