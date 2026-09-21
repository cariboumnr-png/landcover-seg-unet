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

# pylint: disable=missing-function-docstring

'''Unit tests for epoch runner builder (session/engine/epoch/builder.py).'''

# local imports
import landseg.session.engine.epoch as epoch_mod


# ----- `build_epoch_runner` tests
def test_build_epoch_runner_train_eval(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
):
    '''
    Given: Runtime, dataloaders, and dispatcher.
    When: Calling `build_epoch_runner` in 'train_eval' mode.
    Then: Return `EpochRunner` with both trainer and evaluator populated.
    '''
    runner = epoch_mod.build_epoch_runner(
        mock_runtime,
        mock_dataloaders,
        mock_dispatcher,
        mode='train_eval',
    )

    assert isinstance(runner, epoch_mod.EpochRunner)
    assert runner.mode == 'train_eval'
    assert runner.trainer is not None
    assert runner.evaluator is not None


def test_build_epoch_runner_train_only(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
):
    '''
    Given: Runtime, dataloaders, and dispatcher.
    When: Calling `build_epoch_runner` in 'train_only' mode.
    Then: Return `EpochRunner` with trainer populated and evaluator as None.
    '''
    runner = epoch_mod.build_epoch_runner(
        mock_runtime,
        mock_dataloaders,
        mock_dispatcher,
        mode='train_only',
    )

    assert isinstance(runner, epoch_mod.EpochRunner)
    assert runner.mode == 'train_only'
    assert runner.trainer is not None
    assert runner.evaluator is None


def test_build_epoch_runner_eval_only(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
):
    '''
    Given: Runtime, dataloaders, and dispatcher.
    When: Calling `build_epoch_runner` in 'eval_only' mode.
    Then: Return `EpochRunner` with evaluator populated and trainer as None.
    '''
    runner = epoch_mod.build_epoch_runner(
        mock_runtime,
        mock_dataloaders,
        mock_dispatcher,
        mode='eval_only',
    )

    assert isinstance(runner, epoch_mod.EpochRunner)
    assert runner.mode == 'eval_only'
    assert runner.trainer is None
    assert runner.evaluator is not None


def test_build_epoch_runner_with_schedule(
    mock_runtime,
    mock_dataloaders,
    mock_dispatcher,
    session_config,
):
    '''
    Given: Runtime, dataloaders, dispatcher, and custom schedule config.
    When: Calling `build_epoch_runner` with schedule.
    Then: Frequencies are bound to trainer and evaluator.
    '''
    session_config.orchestration.schedule.update_loss_every_n_batch = 10
    session_config.orchestration.schedule.val_every_n_epoch = 2
    session_config.orchestration.schedule.infer_every_n_epoch = 3

    runner = epoch_mod.build_epoch_runner(
        mock_runtime,
        mock_dataloaders,
        mock_dispatcher,
        mode='train_eval',
        schedule=session_config.orchestration.schedule,
    )

    assert runner.trainer.update_every == 10
    assert runner.evaluator.val_every == 2
    assert runner.evaluator.infer_every == 3

