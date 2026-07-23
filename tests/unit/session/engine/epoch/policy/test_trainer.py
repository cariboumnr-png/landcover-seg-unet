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
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

'''Unit tests for trainer policy module (policy/trainer.py).'''

# third-party imports
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
    mock_dispatcher
):
    '''
    Given: Configured `MultiHeadTrainer`.
    When: Executing `train_one_epoch(epoch=1)`.
    Then: Iterates training batches, updates state, and returns `TrainStepResults`.
    '''
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
    assert trainer.state.progress.epoch == 1
    assert trainer.state.progress.global_step == 2
    assert 'on_train_policy_begin' in mock_dispatcher.events
    assert 'on_train_policy_end' in mock_dispatcher.events
    assert 'on_train_batch_end' in mock_dispatcher.events


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
    trainer._head_losses = {'head_1': 2.0}
    trainer._regularization = {}

    trainer.state.batch_cxt.bidx = 1
    trainer._update_training_stats(flush=False)
    assert trainer.results.metrics_updated is False

    trainer._update_training_stats(flush=True)
    assert trainer.results.metrics_updated is True
    assert trainer.results.total_objective == 2.0
