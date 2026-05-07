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
import landseg.session.engine.configs as configs
import landseg.session.engine.optim as optim
import landseg.session.engine.policy as policy
import landseg.session.engine.protocols as protocols
import landseg.session.engine.state as state
import landseg.session.engine.tasks as tasks

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineBuildContext:
    '''Engine building context'''
    dataspecs: core.DataSpecs
    model: core.MultiheadModelLike
    dataloaders: protocols.DataLoadersLike
    batch_size: int
    evaluation_dataset: typing.Literal['val', 'test']
    device: str

# -------------------------------Public Function-------------------------------
def build_engine(
    *,
    context: EngineBuildContext,
    tasks_config: tasks.TaskConfig,
    optim_config: optim.OptimConfig,
    runtime_config: configs.RuntimeConfigLike,
    dispatcher: common.SessionObserverLike,
    mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
) -> engine.EpochRunner:
    '''
    doc
    '''

    runtime_state = state.initialize_state(
        all_heads=list(context.dataspecs.heads.class_counts.keys()),
        batch_size=context.batch_size,
        use_amp=runtime_config.precision.use_amp,
        device=context.device
    )

    # batch engine
    preview_ctx = context.dataloaders.preview_context
    batch_config = batch.BatchExecutorConfig(
        parent_map=context.dataspecs.heads.head_parent,
        use_amp=runtime_config.precision.use_amp,
        patch_per_blk=preview_ctx.patch_per_blk if preview_ctx else None,
        patch_per_dim=preview_ctx.patch_per_dim if preview_ctx else None,
        block_columns=preview_ctx.block_columns if preview_ctx else None
    )
    batch_executor = batch.BatchExecutionEngine(
        context.model,
        runtime_state,
        batch_config,
        device=context.device
    )

    # config model logit adjustment
    batch_executor.model.set_logit_adjust_enabled(runtime_config.logit_adjust.enable)
    batch_executor.model.set_logit_adjust_alpha(runtime_config.logit_adjust.alpha)

    # engine tasks
    engine_tasks = tasks.build_engine_tasks(context.dataspecs, tasks_config)

    # engine optimization
    optimization = optim.build_optimization(context.model, optim_config)

    # trainer
    trainer = policy.MultiHeadTrainer(
        # base engine
        engine=batch_executor,
        engine_state=runtime_state,
        engine_tasks=engine_tasks,
        optimization=optimization,
        dataloaders=context.dataloaders,
        dispatcher=dispatcher,
        device=context.device,
        # trainer-specific
        grad_clip_norm=runtime_config.optimization.grad_clip_norm,
        update_every=runtime_config.schedule.log_loss_every,
    )

    # evaluator
    evaluator = policy.MultiHeadEvaluator(
        # base engine
        engine=batch_executor,
        engine_state=runtime_state,
        engine_tasks=engine_tasks,
        optimization=optimization,
        dataloaders=context.dataloaders,
        dispatcher=dispatcher,
        device=context.device,
        # evaluator-specific
        monitor_heads=runtime_config.monitor.track_heads,
        val_every=runtime_config.schedule.val_every,
        infer_every=runtime_config.schedule.infer_every,
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
