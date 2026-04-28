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
import landseg.session.engine.batch as batch
import landseg.session.engine.policy as policy

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineBuildContext:
    '''Engine building context'''
    dataspecs: core.DataSpecs
    model: core.MultiheadModelLike
    components: common.ComponentsLike
    callbacks: common.CallBacksLike
    runtime_state: common.StateLike

@dataclasses.dataclass
class EngineBuildConfig:
    '''Engine building configs'''
    use_amp: bool
    grad_clip_norm: bool
    loss_update_every: int
    metrics_track_heads: list[str]
    evaluation_dataset: typing.Literal['val', 'test'] = 'val'

# -------------------------------Public Function-------------------------------
def build_engine(
    *,
    context: EngineBuildContext,
    config: EngineBuildConfig,
    mode: typing.Literal['train_evaluate', 'train_only', 'evaluate_only'],
    device: str
) -> policy.EpochRunner:
    '''
    doc
    '''

    # batch engine
    batch_executor = batch.BatchExecutionEngine(
        model=context.model,
        state=context.runtime_state, # type: ignore
        parent_map=context.dataspecs.heads.head_parent,
        use_amp=config.use_amp,
        device=device
    )

    # trainer
    trainer = policy.MultiHeadTrainer(
        engine=batch_executor,
        state=context.runtime_state,
        components=context.components,
        callbacks=context.callbacks,
        device=device,

        grad_clip_norm=config.grad_clip_norm,
        update_every=config.loss_update_every,
    )

    # evaluator
    evaluator = policy.MultiHeadEvaluator(
        engine=batch_executor,
        state=context.runtime_state,
        components=context.components,
        callbacks=context.callbacks,
        device=device,

        monitor_heads=config.metrics_track_heads,
        dataset=config.evaluation_dataset
    )

    # return engine with matched mode
    match mode:
        case 'train_evaluate':
            return policy.EpochRunner(mode, trainer, evaluator)
        case 'train_only':
            return policy.EpochRunner(mode, trainer, None)
        case 'evaluate_only':
            return policy.EpochRunner(mode, None, evaluator)
