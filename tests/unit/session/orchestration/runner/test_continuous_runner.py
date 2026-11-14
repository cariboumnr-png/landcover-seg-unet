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

'''Unit tests for continuous training runner (runner/continuous.py).'''

# local imports
import landseg.session.orchestration.runner.continuous as continuous_mod


def test_continuous_runner_initialization(
    mock_epoch_engine,
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
        epoch_runner=mock_epoch_engine,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    assert runner.phase is phase
    assert runner.epoch_runner is mock_epoch_engine


def test_continuous_runner_run_generator(
    mock_epoch_engine,
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
    mock_save = mocker.patch(
        'landseg.artifacts.save_checkpoint',
        autospec=True
    )
    on_end = mocker.spy(mock_dispatcher, 'on_session_end')
    phase = session_config.orchestration.single_phase
    phase.num_epochs = 2

    runner = continuous_mod.ContinuousRunner(
        phase=phase,
        epoch_runner=mock_epoch_engine,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    steps = list(runner.run())

    assert len(steps) == 2
    assert steps[0].epoch_in_phase == 1
    assert steps[1].epoch_in_phase == 2
    assert steps[1].is_run_end is True

    # assert on_session_end dispatched cleanly
    assert on_end.call_count == 1

    # assert checkpoint recorded valid metric for epoch 1
    first_ckpt_meta = mock_save.call_args_list[0].kwargs['ckpt_meta']
    assert first_ckpt_meta['metric'] != -float('inf')
    assert first_ckpt_meta['metric'] == steps[0].val_metrics_value


def test_continuous_runner_execute(
    mock_epoch_engine,
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
        epoch_runner=mock_epoch_engine,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    metric = runner.execute()

    assert isinstance(metric, float)
