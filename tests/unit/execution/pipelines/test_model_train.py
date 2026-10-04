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
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Unit tests for model train pipeline (model_train.py).
'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.execution.pipelines.model_train as train_pipeline
import landseg.models as models
import landseg.session as session


# ----- `ModelTraining` tests
@pytest.mark.parametrize('mode', ['continuous', 'curriculum'])
def test_train_pipeline_success(tmp_path, dataspecs, monkeypatch, mode: str):
    '''
    Given: A valid RootConfig and mock dependencies for continuous/curriculum.
    When: `ModelTraining.run` is called.
    Then: Execute training runner and record results cleanly.
    '''
    exp_root = str(tmp_path / 'exp')
    prep_root = str(tmp_path / 'prep')
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'SUCCESS'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.data.preparation.output_dpath = prep_root
    schema.session.mode = mode

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    class DummyRunner:
        def execute(self):
            return 0.91

    class DummyModel:
        def parameters(self):
            return []

    mock_model = typing.cast(core.MultiheadModelLike, DummyModel())

    monkeypatch.setattr(
        train_pipeline.geopipe,
        'build_dataspec',
        lambda *a, **kw: dataspecs
    )
    monkeypatch.setattr(
        models,
        'build_multihead_unet',
        lambda *a, **kw: mock_model,
    )
    monkeypatch.setattr(
        session,
        'build_session_runner',
        lambda *a, **kw: DummyRunner(),
    )

    train_pipeline.ModelTraining(config).run()


def test_train_validate_missing_preparation_report(tmp_path):
    '''
    Given: Training configuration without upstream preparation report.
    When: Calling `validate` on `ModelTraining`.
    Then: Raise a RuntimeError indicating missing preparation report.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path / 'exp')
    schema.data.preparation.output_dpath = str(tmp_path / 'prep')

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )
    pipeline = train_pipeline.ModelTraining(config)
    with pytest.raises(
        RuntimeError,
        match='Upstream pipeline "data-prepare" has not been executed yet'
    ):
        pipeline.validate()


def test_train_validate_failed_preparation_report(tmp_path):
    '''
    Given: Training configuration with a failed preparation report.
    When: Calling `validate` on `ModelTraining`.
    Then: Raise a RuntimeError indicating failed preparation status.
    '''
    prep_root = str(tmp_path / 'prep')
    prep_paths = artifacts.PreparationPaths(root=prep_root)
    artifacts.Controller[dict](prep_paths.report).persist({'status': 'FAILED'})

    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path / 'exp')
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
        pipeline.validate()
