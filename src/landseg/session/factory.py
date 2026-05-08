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

Type behavior:
    The session factory uses Literal-based typing and overloads to
    encode execution intent at the type level. This allows static type
    checkers to infer the exact return type of `build_session` based on
    the provided context.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.session.common as common
import landseg.session.data as data
import landseg.session.engine as engine
import landseg.session.instrumentation as instrument
import landseg.session.orchestration as orchestration
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
    def data_loader(self) -> data.DataLoaderConfig: ...
    @property
    def engine_exec(self) -> engine.BatchExecConfigShape: ...
    @property
    def engine_optim(self) -> engine.OptimConfigShape: ...
    @property
    def engine_tasks(self) -> engine.TaskConfigShape: ...
    @property
    def orchestration(self) -> common.OrchestrationConfigShape: ...

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
    device: str
    total_epochs: int
    verbose_runner: bool = True
    eval_dataset: typing.Literal['val', 'test'] = 'val'
    session_paths: artifacts.ResultsPaths | None = None

# -------------------------------Public Function-------------------------------
def build_overfit_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> engine.EpochEngine:
    '''
    doc
    '''

    # callback dispatcher
    dispatcher = instrument.CallbackDispatcher([
        instrument.LoggingCallback(verbose=context.verbose_runner)
    ])
    # context
    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
        logger=logger,
    )
    return engine.build_epoch_engine(
        context=engine_context,
        config=config,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
    )

def build_evaluate_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> engine.EpochEngine:
    '''
    doc
    '''

    # callback dispatcher
    dispatcher = instrument.CallbackDispatcher([
        instrument.LoggingCallback(verbose=context.verbose_runner)
    ])
    # context
    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
        logger=logger,
    )
    return engine.build_epoch_engine(
        context=engine_context,
        config=config,
        mode='eval_only',
        eval_dataset=context.eval_dataset,
    )

def build_continous_training_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> orchestration.ContinuousRunner:
    '''doc'''

    # callback dispatcher
    assert context.session_paths, 'Session paths manager not provided'
    dispatcher = instrument.CallbackDispatcher([
        instrument.LoggingCallback(verbose=context.verbose_runner),
        instrument.TrackingCallback(['tb'], context.session_paths.logs)
    ])
    # epoch engine context
    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
        logger=logger,
    )
    # epoch engine
    epoch_engine = engine.build_epoch_engine(
        context=engine_context,
        config=config,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
    )

    # base orchestrator config
    base_config = orchestration.BaseRunnerConfig(
        artifacts_paths=context.session_paths,
        verbose=context.verbose_runner,
        track_mode=config.orchestration.monitor.track_mode,
        enable_early_stop=config.orchestration.monitor.allow_early_stop,
        patience_epochs=config.orchestration.monitor.patience,
        delta=config.orchestration.monitor.min_delta,
    )

    # return the orchestrator
    return orchestration.build_runner(
        epoch_engine=epoch_engine,
        base_config=base_config,
        runner_type='continuous',
        training_phases=config.orchestration.single_phase,
        dispatcher=dispatcher
    )

def build_curriculum_training_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> orchestration.CurriculumRunner:
    '''doc'''

    # callback dispatcher
    assert context.session_paths, 'Session paths manager not provided'
    dispatcher = instrument.CallbackDispatcher([
        instrument.LoggingCallback(verbose=context.verbose_runner),
        instrument.TrackingCallback(['tb'], context.session_paths.logs)
    ])
    # epoch engine context
    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
        logger=logger,
    )
    # epoch engine
    epoch_engine = engine.build_epoch_engine(
        context=engine_context,
        config=config,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
    )
    # base orchestrator config
    base_config = orchestration.BaseRunnerConfig(
        artifacts_paths=context.session_paths,
        verbose=context.verbose_runner,
        track_mode=config.orchestration.monitor.track_mode,
        enable_early_stop=config.orchestration.monitor.allow_early_stop,
        patience_epochs=config.orchestration.monitor.patience,
        delta=config.orchestration.monitor.min_delta,
    )

    # return the orchestrator
    return orchestration.build_runner(
        epoch_engine=epoch_engine,
        base_config=base_config,
        runner_type='curriculum',
        training_phases=config.orchestration.multi_phases,
        dispatcher=dispatcher
    )
