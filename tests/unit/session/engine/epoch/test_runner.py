# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
import landseg.session.engine.epoch.runner as exec_mod


# ----- `EpochEngine` initialization and total_train_batch tests
def test_epoch_engine_init(mock_trainer, mock_evaluator):
    '''
    Given: Valid trainer and evaluator instances.
    When: Instantiating `EpochEngine` in train_eval mode.
    Then: Store mode, trainer, evaluator and `training_sample_size`.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='train_eval',
        trainer=mock_trainer,
        evaluator=mock_evaluator
    )

    assert epoch_engine.mode == 'train_eval'
    assert epoch_engine.trainer is mock_trainer
    assert epoch_engine.evaluator is mock_evaluator
    assert epoch_engine.training_batch_count == 4 # one block, 4 patches


def test_epoch_engine_train_size_no_trainer(mock_evaluator):
    '''
    Given: `EpochEngine` in eval_only mode with no trainer.
    When: Reading `training_sample_size`.
    Then: Return 0.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='eval_only',
        trainer=None,
        evaluator=mock_evaluator
    )

    assert epoch_engine.training_batch_count == 0


# ----- `run_epoch` execution tests
def test_epoch_engine_run_epoch_train_eval(mock_trainer, mock_evaluator):
    '''
    Given: `EpochEngine` in `train_eval` mode.
    When: Calling `run_epoch(epoch=1)`.
    Then: Execute policies, returning `SessionStepResults`.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='train_eval',
        trainer=mock_trainer,
        evaluator=mock_evaluator
    )
    epoch_engine.set_head_state(active_heads=['head_1'])

    # populate dummy patch entry in infer_out for continuous inference
    dummy_weight = mock_trainer.model.conv.weight.new_zeros(16, 16)
    assert mock_evaluator.state.infer_out is not None
    mock_evaluator.state.infer_out.labels['head_1'] = {(0, 0): dummy_weight}
    mock_evaluator.state.infer_out.preds['head_1'] = {(0, 0): dummy_weight}
    mock_evaluator.state.infer_out.errors['head_1'] = {(0, 0): dummy_weight}

    results = epoch_engine.run_epoch(epoch=1)

    assert isinstance(results, core.SessionStepResults)
    assert results.training is not None
    assert results.validation is not None
    assert results.inference is not None


def test_epoch_engine_run_epoch_train_only(mock_trainer):
    '''
    Given: `EpochEngine` in `train_only` mode.
    When: Calling `run_epoch(epoch=1)`.
    Then: Execute training policy and return training `SessionStepResults`.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='train_only',
        trainer=mock_trainer,
        evaluator=None
    )
    epoch_engine.set_head_state(active_heads=['head_1'])

    results = epoch_engine.run_epoch(epoch=1)

    assert isinstance(results, core.SessionStepResults)
    assert results.training is not None
    assert results.validation is None
    assert results.inference is None


def test_epoch_engine_run_epoch_eval_only(mock_evaluator):
    '''
    Given: `EpochEngine` in `eval_only` mode.
    When: Calling `run_epoch(epoch=1)`.
    Then: Execute validation policy and return validation step results.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='eval_only',
        trainer=None,
        evaluator=mock_evaluator
    )
    epoch_engine.set_head_state(active_heads=['head_1'])

    results = epoch_engine.run_epoch(epoch=1)

    assert isinstance(results, core.SessionStepResults)
    assert results.training is None
    assert results.validation is not None
    assert results.inference is None


# ----- missing components error handling tests
def test_epoch_engine_missing_trainer_raises(mock_evaluator):
    '''
    Given: `EpochEngine` in `train_eval` mode missing trainer.
    When: Calling `run_epoch`.
    Then: Raise `ValueError` matching 'Missing trainer'.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='train_eval',
        trainer=None,  # type: ignore
        evaluator=mock_evaluator
    )

    with pytest.raises(ValueError, match='Missing trainer'):
        epoch_engine.run_epoch(epoch=1)


# ----- head state forwarding tests
def test_epoch_engine_head_state_forwarding(mock_trainer, mock_evaluator):
    '''
    Given: `EpochEngine` with trainer and evaluator.
    When: Calling `set_head_state` and `reset_head_state`.
    Then: Forward configuration to both sub-controllers.
    '''
    epoch_engine = exec_mod.EpochRunner(
        mode='train_eval',
        trainer=mock_trainer,
        evaluator=mock_evaluator
    )

    epoch_engine.set_head_state(active_heads=['head_1'], frozen_heads=['head_2'])

    assert mock_trainer.state.heads.active_heads == ['head_1']
    assert mock_evaluator.state.heads.active_heads == ['head_1']

    epoch_engine.reset_head_state()

    assert mock_trainer.state.heads.active_heads is None
    assert mock_evaluator.state.heads.active_heads is None
