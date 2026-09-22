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
import landseg.session.logger as session_logger
import landseg.session.data as data
import landseg.session.engine as engine
import landseg.session.instrumentation as instrumentation
import landseg.session.orchestration as orchestration_mod


# ----- public types
class SessionConfigShape(typing.Protocol):
    '''Interface for session construction configuration.'''
    @property
    def data_loader(self) -> data.DataLoaderConfig: ...
    @property
    def engine_exec(self) -> engine.BatchExecConfigShape: ...
    @property
    def engine_optim(self) -> engine.OptimConfigShape: ...
    @property
    def engine_tasks(self) -> engine.TaskConfigShape: ...
    @property
    def engine_schedule(self) -> engine.ScheduleConfigShape: ...
    @property
    def orchestration(self) -> orchestration_mod.OrchestrationConfigShape: ...


# ----- public dataclasses
@dataclasses.dataclass
class SessionBuildContext:
    '''Context for session construction.'''
    device: str
    eval_dataset: typing.Literal['val', 'test'] = 'val'
    session_paths: artifacts.SessionPaths | None = None


# ----- public functions
def build_overfit_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: session_logger.SessionLogger | None = None
) -> engine.EpochRunner:
    '''Build an epoch engine for overfit training with evaluation.'''
    # callback dispatcher
    dispatcher = instrumentation.build_dispatcher(
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )
    # dataloaders
    dataloaders = data.build_dataloaders(
        dataspecs,
        config.data_loader,
        logger=logger
    )
    # context
    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
    )
    return engine.build_engine(
        dataloaders,
        engine_context,
        config,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
    )


def build_evaluate_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: session_logger.SessionLogger | None = None
) -> engine.EpochRunner:
    '''Build an epoch engine for evaluation-only execution.'''
    # callback dispatcher
    dispatcher = instrumentation.build_dispatcher(
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )
    # dataloaders
    dataloaders = data.build_dataloaders(
        dataspecs,
        config.data_loader,
        logger=logger
    )
    # context
    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
    )
    return engine.build_engine(
        dataloaders,
        engine_context,
        config,
        mode='eval_only',
        eval_dataset=context.eval_dataset,
    )


def build_continous_training_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: session_logger.SessionLogger | None = None
) -> orchestration_mod.ContinuousRunner:
    '''Build a continuous training runner orchestrator.'''
    assert context.session_paths, 'Session paths manager not provided'
    dispatcher = instrumentation.build_dispatcher(
        trackers=['tb'],
        uri=context.session_paths.logs,
        label_color_map=dataspecs.meta.label_color_map,
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )

    dataloaders = data.build_dataloaders(
        dataspecs,
        config.data_loader,
        logger=logger
    )

    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
    )
    epoch_engine = engine.build_engine(
        dataloaders,
        engine_context,
        config,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
    )

    return orchestration_mod.build_runner(
        epoch_engine,
        config.orchestration,
        dispatcher,
        context.session_paths,
        runner_type='continuous',
    )


def build_curriculum_training_session(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    logger: session_logger.SessionLogger | None = None
) -> orchestration_mod.CurriculumRunner:
    '''Build a multiphase training runner orchestrator.'''
    assert context.session_paths, 'Session paths manager not provided'
    dispatcher = instrumentation.build_dispatcher(
        trackers=['tb'],
        uri=context.session_paths.logs,
        label_color_map=dataspecs.meta.label_color_map,
        logger=logger,
        verbose=(getattr(logger, 'console_lvl', None) is not None)
    )

    dataloaders = data.build_dataloaders(
        dataspecs,
        config.data_loader,
        logger=logger
    )

    engine_context = engine.EpochEngineContext(
        dataspecs=dataspecs,
        model=model,
        dispatcher=dispatcher,
        device=context.device,
    )
    epoch_engine = engine.build_engine(
        dataloaders,
        engine_context,
        config,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
    )

    return orchestration_mod.build_runner(
        epoch_engine,
        config.orchestration,
        dispatcher,
        context.session_paths,
        runner_type='curriculum',
    )
