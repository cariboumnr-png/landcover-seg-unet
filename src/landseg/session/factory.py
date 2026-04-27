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
- evaluator (always instantiated)
- trainer (always instantiated, but may not be used)
- orchestration runner (only in training mode)

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
        phase_schema: Identifier describing how training phases are
            interpreted by the orchestration layer.
        components:
            Configuration used to construct session components such as
            losses, metrics, and dataloaders.
        runtime:
            Runtime configuration controlling precision, optimization,
            scheduling, and monitoring behavior.
        training_phases:
            Ordered sequence of phase definitions used by the training
            orchestration runner.
    '''
    @property
    def phase_schema(self) -> str: ...
    @property
    def components(self) -> comps.ComponentsConfig: ...
    @property
    def runtime(self) -> common.ConfigLike: ...
    @property
    def training_phases(self) -> typing.Sequence[orchestration.PhaseLike]: ...

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
        eval_dataset: Dataset split used for evaluation ('val' or 'test').
        session_paths: Optional artifact paths for saving outputs (
            required for `'training'` mode).
    '''
    intent: typing.Literal['training', 'evaluation', 'overfit']
    device: str
    logger: utils.Logger
    verbose_runner: bool = True
    eval_dataset: typing.Literal['val', 'test'] = 'val'
    session_paths: artifacts.ResultsPaths | None = None

# -------------------------------Public Function-------------------------------
def build_session(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext
) -> tuple[engine.EpochRunner | None, orchestration.TrainingRunner | None]:
    '''
    Assemble the execution pipeline for a multi-head model session.

    This function constructs and wires together all runtime components:
    - session components (losses, metrics, dataloaders)
    - runtime state (device placement, AMP)
    - instrumentation callbacks
    - batch execution engine
    - trainer and evaluator

    The evaluator is always created. The trainer is also instantiated in
    all modes but is only actively used when applicable. The returned
    objects depend on the execution intent.

    Execution modes:
        `'evaluation'`:
            Returns an `EpochRunner` containing only the evaluator.

        `'overfit'`:
            Returns an `EpochRunner` with both trainer and evaluator.
            No orchestration layer is used.

        `'training'`:
            Returns a `TrainingRunner` that manages training phases,
            early stopping, and evaluation. An `EpochRunner` is created
            internally but not returned.

    Args:
        dataspecs: Dataset specifications, including head structure and
            parent relationships.
        model: Multi-head model to execute.
        config: Static session configuration defining components, runtime
            behavior, and training phases.
        context: Runtime context of the execution mode and environment.

    Returns:
        A tuple:
        - `EpochRunner | None`
            Returned in 'evaluation' and 'overfit' modes.
        - `TrainingRunner | None`
            Returned only in 'training' mode.

    Raises:
        AssertionError:
            If `intent='training'` but `session_paths` is not provided.
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
        session_components.headspecs,
        session_components.dataloaders,
        use_amp=config.runtime.precision.use_amp,
        device=context.device
    )

    # build callbacks
    callbacks = instrument.build_callbacks(
        runtime_state, # type: ignore
        config.runtime,
        device=context.device,
    )

    # batch engine
    batch_executor = engine.BatchExecutionEngine(
        model=model,
        state=runtime_state, # type: ignore
        parent_map=dataspecs.heads.head_parent,
        use_amp=config.runtime.precision.use_amp,
        device=context.device
    )

    # trainer
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

    # evaluator
    evaluator = engine.MultiHeadEvaluator(
        engine=batch_executor,
        state=runtime_state,
        components=session_components,
        callbacks=callbacks,
        device=context.device,
        track_mode=config.runtime.monitor.track_mode,
        monitor_heads=config.runtime.monitor.track_heads,
        min_delta=config.runtime.schedule.min_delta,
        dataset=context.eval_dataset
    )

    # build trainer/runner depending on mode
    match context.intent:

        case 'overfit':
            # return an epoch runner with both trainer and evaluator
            return engine.EpochRunner(trainer, evaluator), None

        case 'training':
            # build an epoch runner with both trainer and evaluator
            epoch_runner = engine.EpochRunner(trainer, evaluator)
            # build orchestration runner and return
            assert context.session_paths, 'Session artifacts paths not defined'
            runner_config = orchestration.RunnerConfig(
                artifacts_paths=context.session_paths,
                resume_from_last=False,
                verbose=context.verbose_runner,
                enable_early_stop=True,
                track_mode=config.runtime.monitor.track_mode,
                patience_epochs=config.runtime.schedule.patience,
                delta=config.runtime.schedule.min_delta
            )
            return None, orchestration.TrainingRunner(
                epoch_runner,
                config.training_phases,
                runner_config,
                logger=context.logger,
            )

        case 'evaluation':
            # return an epoch runner with just the evaluator
            return engine.EpochRunner(None, evaluator), None
