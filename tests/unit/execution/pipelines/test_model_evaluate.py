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
import landseg.configs as configs
import landseg.execution.pipelines.model_evaluate as eval_pipeline


# ----- `evaluate` pipeline test
def test_evaluate_invalid_split_raises_value_error(tmp_path):
    '''
    Given: A RootConfig with an invalid evaluation split.
    When: `evaluate` is called.
    Then: Raise a ValueError.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    schema.pipeline.model_evaluate.checkpoint = 'chk.pt'
    schema.pipeline.model_evaluate.split = 'invalid'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    with pytest.raises(ValueError, match='Invalid split'):
        eval_pipeline.evaluate(config)


def test_evaluate_pipeline_success(tmp_path, dataspecs, monkeypatch):
    '''
    Given: A valid RootConfig and mock dependencies.
    When: `evaluate` is called.
    Then: Execute evaluation and persist results.
    '''
    exp_root = str(tmp_path / 'exp')
    chk_file = str(tmp_path / 'chk.pt')
    with open(chk_file, 'w', encoding='utf-8') as f:
        f.write('dummy')

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.pipeline.model_evaluate.checkpoint = chk_file
    schema.pipeline.model_evaluate.split = 'val'

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

    monkeypatch.setattr(
        eval_pipeline.geopipe, 'build_dataspec', lambda *a, **kw: dataspecs
    )
    monkeypatch.setattr(
        eval_pipeline.models, 'build_multihead_unet', lambda *a, **kw: None
    )
    monkeypatch.setattr(
        eval_pipeline.artifacts, 'load_checkpoint', lambda *a, **kw: None
    )
    monkeypatch.setattr(
        eval_pipeline.session,
        'build_session_runner',
        lambda *a, **kw: MockRunner(),
    )

    target_metric = eval_pipeline.evaluate(config)

    assert target_metric == 0.85
    assert os.path.exists(f'{exp_root}/results/run_0001/evaluation.json')
