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
Session-level factory.
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

#
class _SessionConfig(typing.Protocol):
    '''doc'''
    @property
    def components(self) -> comps.ComponentsConfig: ...
    @property
    def runtime(self) -> common.ConfigLike: ...
    @property
    def phases(self) -> typing.Sequence[orchestration.TrainingPhaseLike]: ...

#
@dataclasses.dataclass
class _Session:
    '''doc'''
    evaluator: engine.MultiHeadEvaluator
    trainer: engine.MultiHeadTrainer | None
    training_runner: orchestration.TrainingRunner | None

#
def build_session(
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: _SessionConfig,
    *,
    mode: typing.Literal['train', 'evaluate', 'overfit'],
    device: str,
    logger: utils.Logger,
    skip_callback_logging: bool = False,
    eval_dataset: typing.Literal['val', 'test'] = 'val',
    session_paths: artifacts.ResultsPaths | None = None,
) -> _Session:
    '''
    Build a session.
    '''

    # build session components
    session_components = comps.build_session_components(
        dataspecs,
        model,
        config.components,
        logger=logger,
    )

    # initiate the shared runtime state
    runtime_state = state.initialize(
        session_components,
        use_amp=config.runtime.precision.use_amp,
        device=device
    )

    # build callbacks
    callbacks = instrument.build_callbacks(
        runtime_state, # type: ignore
        config.runtime,
        logger,
        device=device,
        skip_log=skip_callback_logging
    )

    # batch engine
    batch_executor = engine.BatchExecutionEngine(
        model=model,
        state=runtime_state, # type: ignore
        parent_map=dataspecs.heads.head_parent,
        use_amp=config.runtime.precision.use_amp,
        device=device
    )

    # evaluator is always needed
    evaluator = engine.MultiHeadEvaluator(
        engine=batch_executor,
        state=runtime_state,
        components=session_components,
        callbacks=callbacks,
        device=device,
        track_mode=config.runtime.monitor.track_mode,
        track_head_name=config.runtime.monitor.track_head_name,
        min_delta=config.runtime.schedule.min_delta,
        dataset=eval_dataset
    )

    # build trainer/runner depending on mode
    match mode:
        case 'evaluate':
            return _Session(evaluator, None, None)

        case 'train':
            trainer = engine.MultiHeadTrainer(
                engine=batch_executor,
                state=runtime_state,
                components=session_components,
                callbacks=callbacks,
                device=device,
                use_amp=config.runtime.precision.use_amp,
                grad_clip_norm=config.runtime.optimization.grad_clip_norm,
                log_every=config.runtime.schedule.log_every,
            )
            # build controller and return
            assert session_paths, 'Session artifacts paths not defined'
            training_runner = orchestration.TrainingRunner(
                trainer=trainer,
                evaluator=evaluator,
                schedule=config.runtime.schedule,
                phases=config.phases,
                paths=session_paths,
                logger=logger
            )
            return _Session(evaluator, trainer, training_runner)

        case 'overfit':
            trainer = engine.MultiHeadTrainer(
                engine=batch_executor,
                state=runtime_state,
                components=session_components,
                callbacks=callbacks,
                device=device,
                use_amp=config.runtime.precision.use_amp,
                grad_clip_norm=config.runtime.optimization.grad_clip_norm,
                log_every=config.runtime.schedule.log_every,
            )
            return _Session(evaluator, trainer, None)
