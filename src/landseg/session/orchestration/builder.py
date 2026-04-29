# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

'''
Orchestration runner factory.

This module provides a typed factory function for constructing concrete
epoch-based training runners used by the orchestration layer. It is
responsible for selecting between available `BaseRunner` implementations
(e.g. continuous or curriculum-based training) and wiring them with a
shared epoch execution engine, runner configuration, and training phase
definition(s).

The factory performs no orchestration logic itself; it does not execute
epochs, manage policies, or emit training events. Instead, it centralizes
runner instantiation in order to:

- enforce consistent coupling between runner type and phase structure
- provide precise return typing for downstream consumers
- isolate runner selection logic from higher-level CLI / application code

All constructed runners expose training progress exclusively as a
generator of epoch-level ``TrainingStep`` records, ensuring a uniform
external contract regardless of internal training structure.
'''

# standard imports
import typing
# local imports
import landseg.session.common as common
import landseg.session.orchestration.phases as phases
import landseg.session.orchestration.runner as runner
import landseg.utils as utils

@typing.overload
def build_runner(
    *,
    epoch_runner: common.EpochEngineLike,
    base_config: runner.BaseRunnerConfig,
    training_phases: phases.PhaseLike,
    runner_type: typing.Literal['continuous'],
    logger: utils.Logger,
) -> runner.ContinuousRunner: ...

@typing.overload
def build_runner(
    *,
    epoch_runner: common.EpochEngineLike,
    base_config: runner.BaseRunnerConfig,
    training_phases: typing.Sequence[phases.PhaseLike],
    runner_type: typing.Literal['curriculum'],
    logger: utils.Logger,
) -> runner.CurriculumRunner: ...

def build_runner(
    *,
    epoch_runner: common.EpochEngineLike,
    base_config: runner.BaseRunnerConfig,
    training_phases: phases.PhaseLike | typing.Sequence[phases.PhaseLike],
    runner_type: typing.Literal['continuous', 'curriculum'],
    logger: utils.Logger,
) -> runner.BaseRunner:
    '''
    Construct a concrete orchestration runner for epoch-based training.

    This factory selects and instantiates a concrete ``BaseRunner``
    implementation based on ``runner_type``, wiring together a shared
    epoch engine, runner configuration, and training phase definition(s).

    Supported runner types:

    - ``'continuous'``:
      Creates a :class:`runner.ContinuousRunner` that executes a single
      training phase continuously. This mode is intended for simple,
      non-curriculum workflows where one phase defines the entire
      training lifecycle. Progress is exposed as a generator of
      ``TrainingStep`` records, one per completed epoch.

    - ``'curriculum'``:
      Creates a :class:`runner.CurriculumRunner` that executes a sequence
      of training phases (a curriculum) over a shared epoch engine. Each
      phase is run sequentially according to its associated policy, and
      progress is similarly exposed as epoch-level ``TrainingStep``
      records.

    Args:
        epoch_runner: Epoch execution engine responsible for running
            individual epochs. This engine is shared across all phases
            and runners.
        base_config: Configuration common to all runner implementations,
            including artifact handling and output behavior.
        runner_type: Selector for the runner implementation to construct.
            Must be either ``'continuous'`` or ``'curriculum'``.
        training_phases: Training phase definition(s) consumed by the
            runner. For ``'continuous'``, this must be a single
            ``PhaseLike`` instance. For ``'curriculum'``, this must be a
            sequence of ``PhaseLike`` instances executed sequentially.
        start_epoch: Starting epoch index for continuous training. This
            value is only applicable to the ``'continuous'`` runner and
            is ignored for curriculum runners.

    Returns:
        ContinuousRunner or CurriculumRunner
            A concrete runner instance exposing training progress as a
            generator of epoch-level ``TrainingStep`` records.

    Raises:
        ValueError: If ``training_phases`` does not match the
            expected type for the selected ``runner_type``.
    '''

    match runner_type:
        case 'continuous':
            if not isinstance(training_phases, phases.PhaseLike):
                raise ValueError('Continuous training requires a single phase')
            return runner.ContinuousRunner(
                epoch_runner=epoch_runner,
                base_config=base_config,
                phase=training_phases,
                logger=logger,
            )
        case 'curriculum':
            if not (
                isinstance(training_phases, list) and
                all(isinstance(p, phases.PhaseLike) for p in training_phases)
            ):
                raise ValueError('Curriculum expects a sequence of phases')
            return runner.CurriculumRunner(
                epoch_runner=epoch_runner,
                base_config=base_config,
                training_phases=training_phases,
                logger=logger,
            )
