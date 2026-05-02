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
import landseg.session.engine.state as state
import landseg.session.engine.policy as policy

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineBuildContext:
    '''Engine building context'''
    dataspecs: core.DataSpecs
    model: core.MultiheadModelLike
    components: common.ComponentsLike
    device: str

@dataclasses.dataclass
class EngineBuildConfig:
    '''Engine building configs'''
    use_amp: bool
    grad_clip_norm: float | None
    train_update_every_n_batch: int
    val_every_n_epoch: int
    infer_every_n_epoch: int
    metrics_track_heads: list[str]
    evaluation_dataset: typing.Literal['val', 'test'] = 'val'
    enable_logit_adjust: bool = False
    logit_adjust_alpha: float = 1.0

# -------------------------------Public Function-------------------------------
def build_engine(
    *,
    context: EngineBuildContext,
    config: EngineBuildConfig,
    runtime_state: state.EngineState,
    callbacks: typing.Sequence[object],
    mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
) -> engine.EpochRunner:
    '''
    doc
    '''

    # batch engine
    preview_ctx = context.components.dataloaders.preview_context
    batch_config = batch.BatchExecutorConfig(
        parent_map=context.dataspecs.heads.head_parent,
        use_amp=config.use_amp,
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
    batch_executor.model.set_logit_adjust_enabled(config.enable_logit_adjust)
    batch_executor.model.set_logit_adjust_alpha(config.logit_adjust_alpha)

    # trainer
    trainer = policy.MultiHeadTrainer(
        engine=batch_executor,
        engine_state=runtime_state,
        components=context.components,
        callbacks=callbacks,
        device=context.device,
        grad_clip_norm=config.grad_clip_norm,
        update_every=config.train_update_every_n_batch,
    )

    # evaluator
    evaluator = policy.MultiHeadEvaluator(
        engine=batch_executor,
        engine_state=runtime_state,
        components=context.components,
        callbacks=callbacks,
        device=context.device,
        monitor_heads=config.metrics_track_heads,
        dataset=config.evaluation_dataset
    )

    # return engine with matched mode
    match mode:
        case 'train_eval':
            return engine.EpochRunner(mode, trainer, evaluator)
        case 'train_only':
            return engine.EpochRunner(mode, trainer, None)
        case 'eval_only':
            return engine.EpochRunner(mode, None, evaluator)
