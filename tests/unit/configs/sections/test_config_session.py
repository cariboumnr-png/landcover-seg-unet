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
Unit tests for `landseg.configs.schema.session`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.session as session_mod
import landseg.configs.schema.session.dataloader as dataloader_sec
import landseg.configs.schema.session.engine as engine_sec
import landseg.configs.schema.session.evaluation as eval_sec
import landseg.configs.schema.session.orchestration as orchestration_sec


# ----- `SessionConfig` tests
def test_session_config_defaults_and_validation():
    '''
    Given: Default `SessionConfig` instance in continuous mode.
    When: Setting single phase epochs and calling `validate()`.
    Then: Validate default data loader and learning rate settings.
    '''
    session = session_mod.SessionConfig()
    session.orchestration.single_phase.num_epochs = 10
    session.validate()

    assert session.mode == 'continuous'
    assert session.dataloader.patch_size == 128
    assert session.dataloader.batch_size == 16
    assert session.engine_optim.lr == 1e-4


def test_session_config_curriculum_mode_validation():
    '''
    Given: A `SessionConfig` configured in curriculum mode.
    When: `SessionConfig.validate()` runs before/after schema setup.
    Then: Enforce early stop flags and require non-single schemas.
    '''
    session = session_mod.SessionConfig(mode='curriculum')
    # post_init should set allow_early_stop to False for curriculum
    assert session.orchestration.monitor.allow_early_stop is False

    # curriculum mode requires non-single schema
    with pytest.raises(ValueError, match='must not be "single"'):
        session.validate()

    # set valid curriculum schema and epoch count
    session.orchestration.curriculum.schema = 'baseline'
    session.orchestration.curriculum.baseline.phases[0].num_epochs = 10
    session.validate()


def test_session_config_invalid_mode():
    '''
    Given: A `SessionConfig` with an unrecognized mode string.
    When: `SessionConfig.validate()` is executed.
    Then: Raise a ValueError indicating invalid execution mode.
    '''
    session = session_mod.SessionConfig(mode='invalid_mode')
    with pytest.raises(ValueError, match='Invalid mode: invalid_mode'):
        session.validate()


def test_session_subsections_validation():
    '''
    Given: Sub-component configs (`DataLoaderConfig`, `EngineOptim`).
    When: Instantiating and validating with invalid parameters.
    Then: Raise ConfigValidationError for invalid patch sizes or epochs.
    '''
    # data loader validation for negative size
    with pytest.raises(base.ConfigValidationError):
        dataloader_sec.DataLoaderConfig(patch_size=-1).validate()

    # optim config missing T_max for CosAnneal
    with pytest.raises(
        base.ConfigValidationError,
        match='missing T_max for CosAnneal'
    ):
        engine_sec.EngineOptim(
            sched_cls='CosAnneal',
            sched_args={},
        ).validate()

    # tasks config invalid alpha fn
    with pytest.raises(
        base.ConfigValidationError,
        match='Invalid loss alpha function'
    ):
        engine_sec.EngineTask(alpha_fn='invalid_fn').validate()

    # phase start epoch larger than num epochs
    with pytest.raises(base.ConfigValidationError):
        orchestration_sec.PhaseConfig(
            start_epoch=10,
            num_epochs=5,
        ).validate()


# ----- `EvaluationConfig` tests
def test_evaluation_config_validation(tmp_path):
    '''
    Given: `EvaluationConfig` with various checkpoint paths and splits.
    When: `EvaluationConfig.validate()` is called.
    Then: Pass for None or valid files, and raise ConfigValidationError.
    '''
    # default None checkpoint passes
    cfg = eval_sec.EvaluationConfig()
    cfg.validate()

    # existing checkpoint passes
    dummy_ckpt = tmp_path / 'model.pt'
    dummy_ckpt.write_text('dummy')
    eval_sec.EvaluationConfig(checkpoint=str(dummy_ckpt)).validate()

    # missing checkpoint raises
    with pytest.raises(base.ConfigValidationError):
        eval_sec.EvaluationConfig(
            checkpoint=str(tmp_path / 'missing.pt')
        ).validate()

    # invalid split raises
    with pytest.raises(base.ConfigValidationError):
        eval_sec.EvaluationConfig(split='invalid').validate()

