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
Session construction and wiring utilities.

Assembles all components required to execute a multi-head model
workflow, including data loading, runtime execution, task components,
instrumentation, and orchestration runners.

This module defines factory entry points for building session variants
(e.g., overfit, evaluation, continuous training, curriculum training)
from shared configuration and runtime context.

Configuration is split into:

- ``config`` (static):
    Defines *what* to build (components, runtime behavior, scheduling).

- ``context`` (dynamic):
    Defines *how* the session is executed (device, logging, paths).

This separation enables reuse of configuration across execution modes
while keeping invocation-specific concerns explicit.

All public builder functions act as intent-specific entry points that
compose and return the appropriate executable objects.
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
    Configuration interface for session construction.

    Defines the required configuration sections used to assemble all
    session components, including data loading, execution runtime,
    optimization, task definitions, and orchestration behavior.

    This configuration describes the static structure of a session and
    is independent of runtime invocation details.
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
    '''Context for session construction.'''
    device: str
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
    '''Build an epoch engine for overfit training with evaluation.'''

    # callback dispatcher
    dispatcher = instrument.build_dispatcher(
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )
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
    '''Build an epoch engine for evaluation-only execution.'''

    # callback dispatcher
    dispatcher = instrument.build_dispatcher(
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )
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
    '''Build a continuous training runner orchestrator.'''

    # callback dispatcher
    assert context.session_paths, 'Session paths manager not provided'
    dispatcher = instrument.build_dispatcher(
        trackers=['tb'],
        uri=context.session_paths.logs,
        label_color_map=dataspecs.meta.label_color_map,
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )
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
        metric_name=config.orchestration.monitor.metric_name,
        track_heads=config.orchestration.monitor.track_heads,
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
    '''Build a multiphase training runner orchestrator.'''

    # callback dispatcher
    assert context.session_paths, 'Session paths manager not provided'
    dispatcher = instrument.build_dispatcher(
        trackers=['tb'],
        uri=context.session_paths.logs,
        label_color_map=dataspecs.meta.label_color_map,
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )
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
        metric_name=config.orchestration.monitor.metric_name,
        track_heads=config.orchestration.monitor.track_heads,
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
