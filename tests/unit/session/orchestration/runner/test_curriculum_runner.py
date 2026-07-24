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

'''Unit tests for curriculum training runner (runner/curriculum.py).'''

# local imports
import landseg.configs.schema.sections.session as session_schema
import landseg.session.orchestration.runner.curriculum as curriculum_mod


def test_curriculum_runner_initialization(
    dummy_epoch_runner,
    mock_runner_config,
    mock_dispatcher
):
    '''
    Given: Sequence of curriculum training phases.
    When: Instantiating `CurriculumRunner`.
    Then: Store curriculum phases and initialize global epoch tracker.
    '''
    phases = [
        session_schema._Phase(name='phase_1', num_epochs=2),
        session_schema._Phase(name='phase_2', num_epochs=2)
    ]

    runner = curriculum_mod.CurriculumRunner(
        training_phases=phases,
        epoch_runner=dummy_epoch_runner,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    assert len(runner.phases) == 2
    assert runner.phases[0].name == 'phase_1'


def test_curriculum_runner_run_generator(
    dummy_epoch_runner,
    mock_runner_config,
    mock_dispatcher,
    mocker
):
    '''
    Given: CurriculumRunner with 2 sequential phases of 1 epoch each.
    When: Iterating over generator returned by `run()`.
    Then: Execute phases in sequence and yield steps per epoch plus run end.
    '''
    mocker.patch('landseg.artifacts.save_checkpoint', autospec=True)

    phases = [
        session_schema._Phase(name='phase_1', num_epochs=1),
        session_schema._Phase(name='phase_2', num_epochs=1)
    ]

    runner = curriculum_mod.CurriculumRunner(
        training_phases=phases,
        epoch_runner=dummy_epoch_runner,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    steps = list(runner.run())

    assert len(steps) == 2
    assert steps[0].phase_name == 'phase_1'
    assert steps[1].phase_name == 'phase_2'
    assert steps[1].is_run_end is True


def test_curriculum_runner_execute(
    dummy_epoch_runner,
    mock_runner_config,
    mock_dispatcher,
    mocker
):
    '''
    Given: CurriculumRunner with curriculum phases.
    When: Calling `execute()`.
    Then: Persist step results JSON and return final target metric scalar.
    '''
    mocker.patch('landseg.artifacts.save_checkpoint', autospec=True)
    mocker.patch('landseg.artifacts.Controller.persist', autospec=True)

    phases = [session_schema._Phase(name='phase_1', num_epochs=1)]

    runner = curriculum_mod.CurriculumRunner(
        training_phases=phases,
        epoch_runner=dummy_epoch_runner,
        base_config=mock_runner_config,
        dispatcher=mock_dispatcher
    )

    metric = runner.execute()

    assert isinstance(metric, float)
    assert metric == 0.80
