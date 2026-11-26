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

'''Unit tests for trainer policy module (policy/trainer.py).'''

# pylint: disable=duplicate-code
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

# standard imports
import dataclasses
# third-party imports
import pytest
import torch
# local imports
import landseg.core as core
import landseg.session.engine.epoch.policy.trainer as trainer_mod


# ----- `MultiHeadTrainer` initialization tests
def test_trainer_init(mock_runtime, mock_dataloaders, mock_dispatcher):
    '''
    Given: Parameters for `MultiHeadTrainer`.
    When: Instantiating `MultiHeadTrainer`.
    Then: Results container and update interval are set.
    '''
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=2,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

    assert trainer.update_every == 2
    assert isinstance(trainer.results, core.TrainStepResults)


# ----- `train_one_epoch` execution tests
def test_trainer_train_one_epoch(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
    mocker
):
    '''
    Given: Configured `MultiHeadTrainer`.
    When: Executing `train_one_epoch(epoch=1)`.
    Then: Iterate training batches and return `TrainStepResults`.
    '''
    on_begin = mocker.spy(mock_dispatcher, 'on_train_policy_begin')
    on_end = mocker.spy(mock_dispatcher, 'on_train_policy_end')
    on_batch_end = mocker.spy(mock_dispatcher, 'on_train_batch_end')
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    trainer.set_head_state(active_heads=['head_1'])

    results = trainer.train_one_epoch(epoch=1)

    assert isinstance(results, core.TrainStepResults)
    assert trainer.state.progress.epoch == 1        # started from 0
    assert trainer.state.progress.global_step == 1  # started from 0
    on_begin.assert_called_once()
    on_end.assert_called_once_with(results)
    assert on_batch_end.call_count == 1


# ----- `_clip_grad` gradient clipping tests
def test_trainer_clip_grad(mock_runtime, mock_dataloaders, mock_dispatcher):
    '''
    Given: `MultiHeadTrainer` with optimization `grad_clip_norm`.
    When: Executing `_clip_grad()`.
    Then: Call `clip_grad_norm_` without errors.
    '''
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )

    trainer._clip_grad()
    # verify model parameters remain valid finite tensors
    for param in trainer.model.parameters():
        assert torch.isfinite(param).all()


# ----- `_update_training_stats` interval tests
def test_trainer_update_training_stats_interval(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher
):
    '''
    Given: `MultiHeadTrainer` configured with `update_every=2`.
    When: Updating stats on batch index 1 vs flush=True.
    Then: `metrics_updated` flag reflects logging interval.
    '''
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=2,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    trainer._objective = 2.0
    trainer._head_losses = {'head_1': 1.0, 'head_2': 1.0}
    trainer._regularization = {}

    trainer.state.batch_cxt.bidx = 1
    trainer._update_training_stats(flush=False)
    assert trainer.results.metrics_updated is False

    trainer._update_training_stats(flush=True)
    assert trainer.results.metrics_updated is True
    assert trainer.results.total_objective == 2.0


# ----- `MultiHeadTrainer` consecutive skipped steps guard tests
def test_trainer_consecutive_skipped_steps_guard(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
    mocker
):
    '''
    Given: Trainer encountering consecutive unscaled NaN/Inf gradients.
    When: Optimizer step is skipped for 5 consecutive batches.
    Then: Raise `RuntimeError` halting training to prevent corruption.
    '''
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    trainer.set_head_state(active_heads=['head_1'])

    # simulate 5 batches available in dataloader
    single_batch = next(iter(mock_dataloaders.train))
    trainer.dataloaders = dataclasses.replace(
        mock_dataloaders,
        train=[single_batch] * 6
    )

    # simulate GradScaler decreasing scale on every batch
    # scale_before and scale_after are sampled once per batch
    scales = [64.0, 32.0, 32.0, 16.0, 16.0, 8.0, 8.0, 4.0, 4.0, 2.0]
    mocker.patch.object(
        trainer.state.optim.scaler,
        'get_scale',
        side_effect=scales
    )

    with pytest.raises(RuntimeError, match='consecutive batches'):
        trainer.train_one_epoch(epoch=1)


def test_trainer_skipped_steps_resets_on_success(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
    mocker
):
    '''
    Given: Trainer with non-zero consecutive skipped steps counter.
    When: Optimizer step succeeds without scale decrease.
    Then: Reset consecutive skipped steps counter to 0.
    '''
    trainer = trainer_mod.MultiHeadTrainer(
        update_every=1,
        engine_runtime=mock_runtime,
        dataloaders=mock_dataloaders,
        dispatcher=mock_dispatcher,
        device='cpu'
    )
    trainer.set_head_state(active_heads=['head_1'])
    trainer._consecutive_skipped_steps = 3

    # mock get_scale so scale_after >= scale_before
    mocker.patch.object(
        trainer.state.optim.scaler,
        'get_scale',
        side_effect=[32.0, 32.0]
    )

    trainer.train_one_epoch(epoch=1)

    assert trainer._consecutive_skipped_steps == 0
