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
from __future__ import annotations
import dataclasses
import functools
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
    def components(self) -> comps.ComponentsConfig: ...
    @property
    def runtime(self) -> common.ConfigLike: ...
    @property
    def single_phase(self) -> orchestration.PhaseLike: ...
    @property
    def curriculum(self) -> typing.Sequence[orchestration.PhaseLike]: ...

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
) -> engine.EpochRunner:
    '''
    doc
    '''

    epoch_runner_partial = _build_partial_epoch_runner(
        dataspecs,
        model,
        config,
        context,
        logger,
    )
    return epoch_runner_partial(mode='train_eval')

def build_evaluate_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> engine.EpochRunner:
    '''
    doc
    '''

    epoch_runner_partial = _build_partial_epoch_runner(
        dataspecs=dataspecs,
        model=model,
        config=config,
        context=context,
        logger=logger,
    )
    return epoch_runner_partial(mode='eval_only')

def build_continous_training_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> orchestration.ContinuousRunner:
    '''doc'''

    training_runner_partial = _build_partial_training_runner(
        dataspecs=dataspecs,
        model=model,
        config=config,
        context=context,
        logger=logger,
    )
    training_runner = training_runner_partial(
        training_phases=config.single_phase,
        runner_type='continuous',
    )
    return training_runner

def build_curriculum_training_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> orchestration.CurriculumRunner:
    '''doc'''

    training_runner_partial = _build_partial_training_runner(
        dataspecs=dataspecs,
        model=model,
        config=config,
        context=context,
        logger=logger,
    )
    training_runner = training_runner_partial(
        training_phases=config.curriculum,
        runner_type='curriculum',
    )
    return training_runner

def _build_partial_epoch_runner(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> typing.Callable[..., engine.EpochRunner]:
    '''doc'''

    # build session components
    session_components = comps.build_session_components(
        dataspecs,
        model,
        config.components,
        logger=logger,
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

    # build runner context and config
    engine_build_context = engine.EngineBuildContext(
        dataspecs=dataspecs,
        model=model,
        components=session_components,
        callbacks=callbacks,
        runtime_state=runtime_state,
        device=context.device,
    )
    engine_build_config = engine.EngineBuildConfig(
        use_amp=config.runtime.precision.use_amp,
        grad_clip_norm=config.runtime.optimization.grad_clip_norm,
        loss_update_every=config.runtime.schedule.log_every,
        metrics_track_heads=config.runtime.monitor.track_heads,
        evaluation_dataset=context.eval_dataset
    )

    # return a partial epoch runner without the mode flag
    return functools.partial(
        engine.build_engine,
        context=engine_build_context,
        config=engine_build_config,
    )

def _build_partial_training_runner(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: utils.Logger
) -> typing.Callable: # avoid complex typing as it is just an internal wrapper
    '''doc'''

    epoch_runner_partial = _build_partial_epoch_runner(
        dataspecs,
        model,
        config,
        context,
        logger
    )
    epoch_runner = epoch_runner_partial(mode='train_eval')

    assert context.session_paths, 'Session artifacts paths not defined'
    tracking = orchestration.TrackingConfig(
        track_mode=config.runtime.monitor.track_mode,
        enable_early_stop=config.runtime.monitor.allow_early_stop,
        patience_epochs=config.runtime.monitor.patience,
        delta=config.runtime.monitor.min_delta,
    )
    base_config = orchestration.BaseRunnerConfig(
        artifacts_paths=context.session_paths,
        verbose=context.verbose_runner,
        tracking=tracking,
    )

    # return a partial orchestraction runner
    return functools.partial(
        orchestration.build_runner,
        epoch_runner=epoch_runner,
        base_config=base_config,
        logger=logger
    )
