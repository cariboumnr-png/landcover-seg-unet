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

'''
Orchestration runner factory.

This module provides a typed factory function for constructing concrete
epoch-based training runners used by the orchestration layer. It is
responsible for selecting between available `BaseRunner` implementations
(e.g. continuous or curriculum-based training) and wiring them with a
shared epoch execution engine, runner configuration, and training phase
definition(s).

The factory performs no orchestration logic itself; it does not execute
epochs, manage policies, or emit training events. Instead, it
centralizes runner instantiation in order to:

- enforce consistent coupling between runner type and phase structure
- provide precise return typing for downstream consumers
- isolate runner selection logic from higher-level application code

All constructed runners expose training progress exclusively as a
generator of epoch-level ``TrainingStep`` records, ensuring a uniform
external contract regardless of internal training structure.

Public APIs:
    - `build_runner`: factory function constructing concrete runners.
'''

# standard imports
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.session.contracts as contracts
import landseg.session.orchestration.protocols as protocols
import landseg.session.orchestration.runner as runner


# ----- public functions
@typing.overload
def build_runner(
    epoch_engine: protocols.EpochEngineLike,
    config: protocols.OrchestrationConfigShape,
    dispatcher: contracts.SessionObserverLike,
    session_artifact_paths: artifacts.SessionPaths,
    *,
    runner_type: typing.Literal['continuous'],
) -> runner.ContinuousRunner: ...


@typing.overload
def build_runner(
    epoch_engine: protocols.EpochEngineLike,
    config: protocols.OrchestrationConfigShape,
    dispatcher: contracts.SessionObserverLike,
    session_artifact_paths: artifacts.SessionPaths,
    *,
    runner_type: typing.Literal['curriculum'],
) -> runner.CurriculumRunner: ...


def build_runner(
    epoch_engine: protocols.EpochEngineLike,
    config: protocols.OrchestrationConfigShape,
    dispatcher: contracts.SessionObserverLike,
    session_artifact_paths: artifacts.SessionPaths,
    *,
    runner_type: typing.Literal['continuous', 'curriculum'],
) -> runner.ContinuousRunner | runner.CurriculumRunner:
    '''
    Construct a concrete orchestration runner for epoch-based training.

    This factory selects and instantiates a concrete `BaseRunner`
    implementation based on `runner_type`, wiring together a shared
    epoch engine, runner configuration, and training phase
    definition(s).

    Supported runner types:

    - `'continuous'`:
      Creates a `ContinuousRunner` that executes a single training phase
      continuously. This mode is intended for simple workflows where one
      phase defines the entire training lifecycle.

    - `'curriculum'`:
      Creates a `CurriculumRunner` that executes a sequence of training
      phases (a curriculum) over a shared epoch engine. Each phase is
      run sequentially according to its associated policy.

    Args:
        epoch_engine:
            epoch execution engine responsible for running epochs.
        config:
            orchestration configuration containing monitor settings and
            phase definitions.
        dispatcher:
            callback dispatcher broadcasting lifecycle events to
            registered observers.
        session_artifact_paths:
            resolved session directory paths for saving checkpoints and
            metrics.
        runner_type:
            selector for the runner implementation to construct
            (`'continuous'` or `'curriculum'`).

    Returns:
        runner.ContinuousRunner | runner.CurriculumRunner:
            concrete runner instance exposing training progress as a
            generator of step records.

    Raises:
        ValueError:
            if phase configuration does not match the expected type
            for the selected `runner_type`.
    '''
    base_config = runner.BaseRunnerConfig(
        artifacts_paths=session_artifact_paths,
        metric_name=config.monitor.metric_name,
        track_heads=config.monitor.track_heads,
        track_mode=config.monitor.track_mode,
        enable_early_stop=config.monitor.allow_early_stop,
        patience_epochs=config.monitor.patience,
        delta=config.monitor.min_delta,
    )

    match runner_type:
        case 'continuous':
            training_phases = config.single_phase
            if not isinstance(training_phases, contracts.PhaseLike):
                raise ValueError('Continuous training requires a single phase')
            return runner.ContinuousRunner(
                epoch_runner=epoch_engine,
                base_config=base_config,
                dispatcher=dispatcher,
                phase=training_phases,
            )
        case 'curriculum':
            training_phases = config.multi_phases
            if not (
                isinstance(training_phases, list) and
                all(isinstance(p, contracts.PhaseLike) for p in training_phases)
            ):
                raise ValueError('Curriculum expects a sequence of phases')
            return runner.CurriculumRunner(
                epoch_runner=epoch_engine,
                base_config=base_config,
                dispatcher=dispatcher,
                training_phases=training_phases,
            )
