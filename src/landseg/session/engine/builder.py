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
Engine building
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.engine as engine
import landseg.session.engine.batch as batch
import landseg.session.engine.optim as optim
import landseg.session.engine.policy as policy
import landseg.session.engine.protocols as protocols
import landseg.session.engine.state as state
import landseg.session.engine.tasks as tasks

# ---------------------------------Public Type---------------------------------
class EngineBuildConfigShape(typing.Protocol):
    '''Structural typing interface for engine building.'''
    @property
    def tasks(self) -> tasks.TaskConfig: ...
    @property
    def optimization(self) -> optim.OptimConfig: ...
    @property
    def runtime(self) -> common.RuntimeConfigLike: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineBuildContext:
    '''Engine building context'''
    dataspecs: core.DataSpecs
    model: core.MultiheadModelLike
    dataloaders: protocols.DataLoadersLike
    evaluation_dataset: typing.Literal['val', 'test']
    device: str

# -------------------------------Public Function-------------------------------
def build_engine(
    *,
    context: EngineBuildContext,
    config: EngineBuildConfigShape,
    dispatcher: common.SessionObserverLike,
    mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
) -> engine.EpochRunner:
    '''
    doc
    '''

    # aliases
    runtime = config.runtime
    tasks_config = config.tasks
    optim_config = config.optimization

    # initialize engine state
    engine_state = state.initialize_state(
        all_heads=list(context.dataspecs.heads.class_counts.keys()),
        batch_size=context.dataloaders.batch_size,
        use_amp=runtime.precision.use_amp,
        device=context.device
    )

    # batch engine
    preview_ctx = context.dataloaders.preview_context
    batch_config = batch.BatchExecutorConfig(
        parent_map=context.dataspecs.heads.head_parent,
        use_amp=runtime.precision.use_amp,
        patch_per_blk=preview_ctx.patch_per_blk if preview_ctx else None,
        patch_per_dim=preview_ctx.patch_per_dim if preview_ctx else None,
        block_columns=preview_ctx.block_columns if preview_ctx else None
    )
    batch_executor = batch.BatchExecutionEngine(
        context.model,
        engine_state,
        batch_config,
        device=context.device
    )

    # config model logit adjustment
    batch_executor.model.set_logit_adjust_enabled(runtime.logit_adjust.enable)
    batch_executor.model.set_logit_adjust_alpha(runtime.logit_adjust.alpha)

    # engine tasks
    engine_tasks = tasks.build_engine_tasks(context.dataspecs, tasks_config)

    # engine optimization
    optimization = optim.build_optimization(context.model, optim_config)

    # engine core bundle
    engine_core = policy.EngineCore(
        engine=batch_executor,
        engine_state=engine_state,
        engine_tasks=engine_tasks,
        engine_optim=optimization
    )

    # trainer
    trainer = policy.MultiHeadTrainer(
        # base engine
        engine_core=engine_core,
        dataloaders=context.dataloaders,
        dispatcher=dispatcher,
        device=context.device,
        # trainer-specific
        grad_clip_norm=runtime.optimization.grad_clip_norm,
        update_every=runtime.schedule.log_loss_every,
    )

    # evaluator
    evaluator = policy.MultiHeadEvaluator(
        # base engine
        engine_core=engine_core,
        dataloaders=context.dataloaders,
        dispatcher=dispatcher,
        device=context.device,
        # evaluator-specific
        val_every=runtime.schedule.val_every,
        infer_every=runtime.schedule.infer_every,
        dataset=context.evaluation_dataset,
    )

    # return engine with matched mode
    match mode:
        case 'train_eval':
            return engine.EpochRunner(mode, trainer, evaluator)
        case 'train_only':
            return engine.EpochRunner(mode, trainer, None)
        case 'eval_only':
            return engine.EpochRunner(mode, None, evaluator)
