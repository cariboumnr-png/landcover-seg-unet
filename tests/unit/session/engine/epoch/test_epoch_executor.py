# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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

# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

'''Unit tests for epoch executor module (executor.py).'''

# third-party imports
import pytest
# local imports
import landseg.core as core
import landseg.session.engine.epoch.executor as exec_mod
import landseg.session.engine.epoch.policy.evaluator as eval_mod
import landseg.session.engine.epoch.policy.trainer as trainer_mod


# ----- fixture helpers for EpochEngine testing
@pytest.fixture
def trainer_and_evaluator(mock_runtime, mock_dataloaders, mock_dispatcher):
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    evaluator = eval_mod.MultiHeadEvaluator(
        val_every=1,
        infer_every=1,
        dataset='val',
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    return trainer, evaluator


# ----- `EpochEngine` initialization and total_train_batch tests
def test_epoch_engine_init(trainer_and_evaluator):
    '''
    Given: Valid trainer and evaluator instances.
    When: Instantiating `EpochEngine` in train_eval mode.
    Then: Store mode, trainer, evaluator and compute total_train_batch.
    '''
    trainer, evaluator = trainer_and_evaluator

    epoch_engine = exec_mod.EpochEngine(
        mode='train_eval',
        trainer=trainer,
        evaluator=evaluator
    )

    assert epoch_engine.mode == 'train_eval'
    assert epoch_engine.trainer is trainer
    assert epoch_engine.evaluator is evaluator
    assert epoch_engine.total_train_batch == 4  # 2 batches * batch_size 2


def test_epoch_engine_total_train_batch_no_trainer(trainer_and_evaluator):
    '''
    Given: `EpochEngine` in eval_only mode with no trainer.
    When: Reading `total_train_batch`.
    Then: Return 0.
    '''
    _, evaluator = trainer_and_evaluator

    epoch_engine = exec_mod.EpochEngine(
        mode='eval_only',
        trainer=None,
        evaluator=evaluator
    )

    assert epoch_engine.total_train_batch == 0


# ----- `run_epoch` execution tests
def test_epoch_engine_run_epoch_train_eval(trainer_and_evaluator):
    '''
    Given: `EpochEngine` in `train_eval` mode.
    When: Calling `run_epoch(epoch=1)`.
    Then: Execute train, val, and infer policies, returning `SessionStepResults`.
    '''
    trainer, evaluator = trainer_and_evaluator
    epoch_engine = exec_mod.EpochEngine(
        mode='train_eval',
        trainer=trainer,
        evaluator=evaluator
    )
    epoch_engine.set_head_state(active_heads=['head_1'])

    # populate dummy patch entry in infer_out for continuous inference
    dummy_weight = trainer.model.conv.weight.new_zeros(16, 16)
    evaluator.state.infer_out.labels['head_1'] = {(0, 0): dummy_weight}
    evaluator.state.infer_out.preds['head_1'] = {(0, 0): dummy_weight}
    evaluator.state.infer_out.errors['head_1'] = {(0, 0): dummy_weight}

    results = epoch_engine.run_epoch(epoch=1)

    assert isinstance(results, core.SessionStepResults)
    assert results.training is not None
    assert results.validation is not None
    assert results.inference is not None


def test_epoch_engine_run_epoch_train_only(trainer_and_evaluator):
    '''
    Given: `EpochEngine` in `train_only` mode.
    When: Calling `run_epoch(epoch=1)`.
    Then: Execute training policy and return `SessionStepResults(training, None, None)`.
    '''
    trainer, _ = trainer_and_evaluator
    epoch_engine = exec_mod.EpochEngine(
        mode='train_only',
        trainer=trainer,
        evaluator=None
    )
    epoch_engine.set_head_state(active_heads=['head_1'])

    results = epoch_engine.run_epoch(epoch=1)

    assert isinstance(results, core.SessionStepResults)
    assert results.training is not None
    assert results.validation is None
    assert results.inference is None


def test_epoch_engine_run_epoch_eval_only(trainer_and_evaluator):
    '''
    Given: `EpochEngine` in `eval_only` mode.
    When: Calling `run_epoch(epoch=1)`.
    Then: Execute validation policy and return `SessionStepResults(None, validation, None)`.
    '''
    _, evaluator = trainer_and_evaluator
    epoch_engine = exec_mod.EpochEngine(
        mode='eval_only',
        trainer=None,
        evaluator=evaluator
    )
    epoch_engine.set_head_state(active_heads=['head_1'])

    results = epoch_engine.run_epoch(epoch=1)

    assert isinstance(results, core.SessionStepResults)
    assert results.training is None
    assert results.validation is not None
    assert results.inference is None


# ----- missing components error handling tests
def test_epoch_engine_missing_trainer_raises(trainer_and_evaluator):
    '''
    Given: `EpochEngine` in `train_eval` mode missing trainer.
    When: Calling `run_epoch`.
    Then: Raise `ValueError` matching 'Missing trainer'.
    '''
    _, evaluator = trainer_and_evaluator
    epoch_engine = exec_mod.EpochEngine(
        mode='train_eval',
        trainer=None,  # type: ignore
        evaluator=evaluator
    )

    with pytest.raises(ValueError, match='Missing trainer'):
        epoch_engine.run_epoch(epoch=1)


# ----- head state forwarding tests
def test_epoch_engine_head_state_forwarding(trainer_and_evaluator):
    '''
    Given: `EpochEngine` with trainer and evaluator.
    When: Calling `set_head_state` and `reset_head_state`.
    Then: Forward configuration to both sub-controllers.
    '''
    trainer, evaluator = trainer_and_evaluator
    epoch_engine = exec_mod.EpochEngine(
        mode='train_eval',
        trainer=trainer,
        evaluator=evaluator
    )

    epoch_engine.set_head_state(active_heads=['head_1'], frozen_heads=['head_2'])

    assert trainer.state.heads.active_heads == ['head_1']
    assert evaluator.state.heads.active_heads == ['head_1']

    epoch_engine.reset_head_state()

    assert trainer.state.heads.active_heads is None
    assert evaluator.state.heads.active_heads is None
