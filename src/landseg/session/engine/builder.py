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
import landseg.session.engine.policy as policy
import landseg.session.engine.protocols as protocols

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
    config: batch.EngineBuildConfigShape,
    dispatcher: common.SessionObserverLike,
    mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
) -> engine.EpochRunner:
    '''
    doc
    '''

    engine_core = batch.build_engine_core(
        dataspecs=context.dataspecs,
        dataloaders=context.dataloaders,
        model=context.model,
        config=config,
        device=context.device
    )

    # trainer
    trainer = policy.MultiHeadTrainer(
        # base engine
        engine_core=engine_core,
        dataloaders=context.dataloaders,
        dispatcher=dispatcher,
        device=context.device,
        # trainer-specific
        grad_clip_norm=config.runtime.optimization.grad_clip_norm,
        update_every=config.runtime.schedule.log_loss_every,
    )

    # evaluator
    evaluator = policy.MultiHeadEvaluator(
        # base engine
        engine_core=engine_core,
        dataloaders=context.dataloaders,
        dispatcher=dispatcher,
        device=context.device,
        # evaluator-specific
        val_every=config.runtime.schedule.val_every,
        infer_every=config.runtime.schedule.infer_every,
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
