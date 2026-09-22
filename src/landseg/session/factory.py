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
import landseg.session.engine as engine_mod
import landseg.session.instrumentation as instrumentation
import landseg.session.orchestration as orchestration_mod


# ----- typing alises
SessionType = typing.Literal[
    'overfit',
    'evaluate',
    'continuous',
    'curriculum',
]


# ----- public types
class SessionConfigShape(typing.Protocol):
    '''Interface for session construction configuration.'''
    @property
    def dataloader(self) -> data.DataLoaderConfig: ...
    @property
    def engine(self) -> engine_mod.EngineConfigShape: ...
    @property
    def orchestration(self) -> orchestration_mod.OrchestrationConfigShape: ...


# ----- public dataclasses
@dataclasses.dataclass
class SessionBuildContext:
    '''Context for session construction.'''
    device: str
    session_paths: artifacts.SessionPaths
    eval_dataset: typing.Literal['val', 'test'] = 'val'
    logger: session_logger.SessionLogger | None = None


# ----- public functions
@typing.overload
def build_session_runner(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    session_type: typing.Literal['overfit', 'evaluate'],
) -> engine_mod.EpochRunner: ...


@typing.overload
def build_session_runner(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    session_type: typing.Literal['continuous'],
) -> orchestration_mod.ContinuousRunner: ...


@typing.overload
def build_session_runner(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    session_type: typing.Literal['curriculum'],
) -> orchestration_mod.CurriculumRunner: ...


def build_session_runner(
    *,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: SessionConfigShape,
    context: SessionBuildContext,
    session_type: SessionType,
) -> (
    engine_mod.EpochRunner |
    orchestration_mod.ContinuousRunner |
    orchestration_mod.CurriculumRunner
):
    '''Build a configured session variant.'''
    logger = context.logger

    if session_type in ['overfit', 'evaluate']:
        dispatcher = instrumentation.build_dispatcher(logger=logger)
        return engine_mod.build_engine(
            _get_engine_context(dataspecs, model, dispatcher, config, logger),
            config.engine,
            mode='eval_only' if session_type == 'evaluate' else 'train_eval',
            eval_dataset=context.eval_dataset,
            device=context.device,
        )

    dispatcher = instrumentation.build_dispatcher(
        trackers=['tb'],
        uri=context.session_paths.logs,
        label_color_map=dataspecs.meta.label_color_map,
        logger=logger,
    )
    epoch_engine = engine_mod.build_engine(
        _get_engine_context(dataspecs, model, dispatcher, config, logger),
        config.engine,
        mode='train_eval',
        eval_dataset=context.eval_dataset,
        device=context.device,
    )

    if session_type == 'continuous':
        return orchestration_mod.build_runner(
        epoch_engine,
        config.orchestration,
        dispatcher,
        context.session_paths,
        runner_type='continuous',
    )

    return orchestration_mod.build_runner(
        epoch_engine,
        config.orchestration,
        dispatcher,
        context.session_paths,
        runner_type='curriculum',
    )


# ----- private helpers
def _get_engine_context(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    dispatcher: instrumentation.CallbackDispatcher,
    config: SessionConfigShape,
    logger: session_logger.SessionLogger | None = None
) -> engine_mod.EngineContext:
    '''Simple helper to return `EngineContext`.'''
    dataloaders = data.build_dataloaders(
        dataspecs,
        config.dataloader,
        logger=logger
    )
    return engine_mod.EngineContext(
        dataspecs=dataspecs,
        model=model,
        dataloaders=dataloaders,
        dispatcher=dispatcher,
    )
