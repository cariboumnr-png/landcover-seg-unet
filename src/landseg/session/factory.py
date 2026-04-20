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

# pylint: disable=missing-function-docstring

'''
Session construction and wiring.

This module provides a factory (`build_session`) that assembles all
runtime objects required to execute a multi-head model workflow,
including:
- component graph (losses, metrics, etc.)
- runtime state (device placement, AMP configuration)
- instrumentation callbacks
- batch execution engine
- evaluator (always present)
- trainer and orchestration runner (conditional)

Configuration is split into two distinct inputs:

- `config` (static, user-defined):
  Structured configuration originating from external sources (e.g., Hydra
  dataclasses). It defines *what* to build: components, runtime behavior,
  and training phases.

- `context` (dynamic, invocation-time):
  Execution-specific parameters that define *how* the session is run in a
  given invocation. This includes intent (training/evaluation/overfit),
  device selection, logging, and runtime flags.

This separation allows the same configuration to be reused across
different execution modes without modification, while keeping runtime
concerns explicit and localized.

The entry point is `build_session`, which returns a `SessionExcutables`
container with the constructed evaluator and optional training components
depending on the selected intent.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.session.common as common
import landseg.session.components as comps
import landseg.session.engine as engine
import landseg.session.instrumentation as instrument
import landseg.session.orchestration as orchestration
import landseg.session.state as state
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
class SessionConfigShape(typing.Protocol):
    '''
    Structural typing interface for session configuration.

    Defines the minimum required configuration attributes used to build
    a session.

    Attributes:
        components: Configuration for constructing session components
            (e.g., losses, metrics).
        runtime: Runtime configuration (precision, scheduling, monitoring,
            optimization).
        phases: Ordered sequence of training phases defining the training
            schedule.
    '''
    @property
    def components(self) -> comps.ComponentsConfig: ...
    @property
    def runtime(self) -> common.ConfigLike: ...
    @property
    def phases(self) -> typing.Sequence[orchestration.TrainingPhaseLike]: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class SessionBuildContext:
    '''
    Context object providing runtime parameters for session construction.

    Attributes:
        intent: Execution mode for the session:
            - 'training': full training loop with evaluation and
                scheduling
            - 'evaluation': evaluation-only mode
            - 'overfit': training without orchestration (typically for
                debugging)
        device: Target compute device (e.g., 'cpu', 'cuda').
        logger: Logger instance for runtime logging and instrumentation.
        skip_callback_logging: If True, disables logging within callbacks.
        eval_dataset: Dataset split used for evaluation ('val' or 'test').
        session_paths: Optional artifact paths for saving outputs (
            required for training).
    '''
    intent: typing.Literal['training', 'evaluation', 'overfit']
    device: str
    logger: utils.Logger
    skip_callback_logging: bool = False
    eval_dataset: typing.Literal['val', 'test'] = 'val'
    session_paths: artifacts.ResultsPaths | None = None

@dataclasses.dataclass
class SessionExecutables:
    '''
    Container for executable session components.

    Attributes:
        evaluator: Evaluation engine for multi-head models.
        trainer: Training engine (None if not applicable).
        training_runner: Orchestration runner managing training phases
            (None if not applicable).
    '''
    evaluator: engine.MultiHeadEvaluator
    trainer: engine.MultiHeadTrainer | None
    training_runner: orchestration.TrainingRunner | None

# -------------------------------Public Function-------------------------------
def build_session(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext
) -> SessionExecutables:
    '''
    Construct and initialize a session execution pipeline.

    This function assembles all required components for model execution,
    including:
    - Session components (losses, metrics, etc.)
    - Runtime state (device placement, AMP configuration)
    - Instrumentation callbacks
    - Batch execution engine
    - Evaluator (always created)
    - Trainer and orchestration runner (conditionally created based on
        intent)

    Behavior varies based on `context.intent`:
        - 'evaluation': builds evaluator only
        - 'training': builds evaluator, trainer, and training runner
        - 'overfit': builds evaluator and trainer without orchestration

    Args:
        dataspecs: Dataset specifications including head structure and
            relationships.
        model: Multi-head model instance to be executed.
        config: Session configuration conforming to SessionConfigShape.
        context: Build-time context specifying execution mode and runtime
            settings.

    Returns:
        SessionExcutables: Container with evaluator and optional
            trainer/runner.

    Raises:
        AssertionError: If `intent='training'` and `session_paths` is not
            provided.
    '''

    # build session components
    session_components = comps.build_session_components(
        dataspecs,
        model,
        config.components,
        logger=context.logger,
    )

    # initiate the shared runtime state
    runtime_state = state.initialize(
        session_components,
        use_amp=config.runtime.precision.use_amp,
        device=context.device
    )

    # build callbacks
    callbacks = instrument.build_callbacks(
        runtime_state, # type: ignore
        config.runtime,
        context.logger,
        device=context.device,
        skip_log=context.skip_callback_logging
    )

    # batch engine
    batch_executor = engine.BatchExecutionEngine(
        model=model,
        state=runtime_state, # type: ignore
        parent_map=dataspecs.heads.head_parent,
        use_amp=config.runtime.precision.use_amp,
        device=context.device
    )

    # evaluator is always needed
    evaluator = engine.MultiHeadEvaluator(
        engine=batch_executor,
        state=runtime_state,
        components=session_components,
        callbacks=callbacks,
        device=context.device,
        track_mode=config.runtime.monitor.track_mode,
        track_head_name=config.runtime.monitor.track_head_name,
        min_delta=config.runtime.schedule.min_delta,
        dataset=context.eval_dataset
    )

    # build trainer/runner depending on mode
    match context.intent:
        case 'evaluation':
            return SessionExecutables(evaluator, None, None)

        case 'training':
            trainer = engine.MultiHeadTrainer(
                engine=batch_executor,
                state=runtime_state,
                components=session_components,
                callbacks=callbacks,
                device=context.device,
                use_amp=config.runtime.precision.use_amp,
                grad_clip_norm=config.runtime.optimization.grad_clip_norm,
                log_every=config.runtime.schedule.log_every,
            )
            # build controller and return
            assert context.session_paths, 'Session artifacts paths not defined'
            training_runner = orchestration.TrainingRunner(
                trainer=trainer,
                evaluator=evaluator,
                schedule=config.runtime.schedule,
                phases=config.phases,
                paths=context.session_paths,
                logger=context.logger
            )
            return SessionExecutables(evaluator, trainer, training_runner)

        case 'overfit':
            trainer = engine.MultiHeadTrainer(
                engine=batch_executor,
                state=runtime_state,
                components=session_components,
                callbacks=callbacks,
                device=context.device,
                use_amp=config.runtime.precision.use_amp,
                grad_clip_norm=config.runtime.optimization.grad_clip_norm,
                log_every=config.runtime.schedule.log_every,
            )
            return SessionExecutables(evaluator, trainer, None)
