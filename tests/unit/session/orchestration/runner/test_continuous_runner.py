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

# pylint: disable=protected-access

'''Unit tests for continuous training runner (runner/continuous.py).'''

# local imports
import landseg.session.orchestration.runner.continuous as continuous_mod


def test_continuous_runner_initialization(
    dummy_epoch_runner,
    mock_runner_config,
    mock_dispatcher,
    session_config
):
    '''
    Given: Valid epoch runner, base config, and dispatcher.
    When: Instantiating `ContinuousRunner`.
    Then: Runner is properly initialized with given phase.
    '''
    phase = session_config.orchestration.single_phase
    phase.num_epochs = 2

    runner = continuous_mod.ContinuousRunner(
        phase=phase,
        epoch_runner=dummy_epoch_runner,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    assert runner.phase is phase
    assert runner.epoch_runner is dummy_epoch_runner


def test_continuous_runner_run_generator(
    dummy_epoch_runner,
    mock_runner_config,
    mock_dispatcher,
    session_config,
    mocker
):
    '''
    Given: ContinuousRunner with 2-epoch phase.
    When: Iterating over generator returned by `run()`.
    Then: Yield `TrainingStep` per epoch plus final terminal step.
    '''
    mocker.patch(
        'landseg.artifacts.save_checkpoint',
        autospec=True
    )
    phase = session_config.orchestration.single_phase
    phase.num_epochs = 2

    runner = continuous_mod.ContinuousRunner(
        phase=phase,
        epoch_runner=dummy_epoch_runner,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    steps = list(runner.run())

    assert len(steps) == 2
    assert steps[0].epoch_in_phase == 1
    assert steps[1].epoch_in_phase == 2
    assert steps[1].is_run_end is True


def test_continuous_runner_execute(
    dummy_epoch_runner,
    mock_runner_config,
    mock_dispatcher,
    session_config,
    mocker
):
    '''
    Given: ContinuousRunner configured for single phase execution.
    When: Calling `execute()`.
    Then: Persist step results JSON and return final scalar target metric.
    '''
    mocker.patch('landseg.artifacts.save_checkpoint', autospec=True)
    mocker.patch('landseg.artifacts.Controller.persist', autospec=True)

    phase = session_config.orchestration.single_phase
    phase.num_epochs = 1

    runner = continuous_mod.ContinuousRunner(
        phase=phase,
        epoch_runner=dummy_epoch_runner,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    metric = runner.execute()

    assert isinstance(metric, float)
    assert metric == 0.80
