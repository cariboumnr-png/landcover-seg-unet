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
Epoch engine construction utilities.

Builds the epoch-level execution engine by assembling data loaders,
runtime execution components, and training/evaluation policies from
dataset metadata and configuration.

This module serves as the orchestration entry point that wires together
all components required for epoch-wise training and evaluation.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.data as data
import landseg.session.engine.epoch as epoch
import landseg.session.engine.runtime.executor as executor
import landseg.session.engine.runtime.optim as optim
import landseg.session.engine.runtime.tasks as tasks
import landseg.session.engine.runtime as runtime
import landseg.session.instrumentation as instrument
import landseg.utils as utils

# --------------------------------priavte  type--------------------------------
class _EpochEngineConfigShape(typing.Protocol):
    '''
    Configuration interface for constructing the epoch engine.

    Defines the required configuration sections used to build data
    loaders, execution runtime, optimization, task components, and
    orchestration scheduling behavior.
    '''
    @property
    def data_loader(self) -> data.DataLoaderConfig: ...
    @property
    def engine_exec(self) -> executor.BatchExecConfigShape: ...
    @property
    def engine_optim(self) -> optim.OptimConfigShape: ...
    @property
    def engine_tasks(self) -> tasks.TaskConfigShape: ...
    @property
    def orchestration(self) -> common.OrchestrationConfigShape: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EpochEngineContext:
    '''Runtime context required for building the epoch engine.'''
    dataspecs: core.DataSpecs
    model: core.MultiheadModelLike
    dispatcher: instrument.CallbackDispatcher
    device: str
    logger: utils.Logger

# -------------------------------Public Function-------------------------------
def build_epoch_engine(
    *,
    context: EpochEngineContext,
    config: _EpochEngineConfigShape,
    mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
    eval_dataset: typing.Literal['val', 'test'] = 'val'
) -> epoch.EpochEngine:
    '''
    Construct an epoch engine with training and/or evaluation policies.

    Assembles all required components, including dataloaders, execution
    runtime, trainer, and evaluator, and returns an ``EpochEngine``
    configured for the specified mode.

    Args:
        context: Runtime context containing dataset specs, model,
            dispatcher, device, and logger.
        config: Configuration providing data loading, execution,
            optimization, task, and orchestration settings.
        mode: Execution mode determining which policies are active:
            - `'train_eval'`: both training and evaluation
            - `'train_only'`: training only
            - `'eval_only'`: evaluation only
        eval_dataset:
            Dataset split used for evaluation (`'val'` or `'test'`).

    Returns:
        epoch.EpochEngine:
            Fully constructed epoch engine with appropriate policies.

    Notes:
        - Trainer and evaluator share the same execution runtime.
        - Scheduling behavior is controlled via orchestration config.
        - Components are assembled once and reused across epochs.
    '''

    # data loader
    data_loaders = data.build_dataloaders(
        context.dataspecs,
        config.data_loader,
        logger=context.logger
    )

    # engine runtime
    engine_runtime = runtime.build_engine_runtime(
        dataspecs=context.dataspecs,
        dataloaders=data_loaders,
        model=context.model,
        config=config,
        device=context.device
    )

    # trainer
    trainer = epoch.MultiHeadTrainer(
        # base engine
        engine_runtime=engine_runtime,
        dataloaders=data_loaders,
        dispatcher=context.dispatcher,
        device=context.device,
        # trainer-specific
        update_every=config.orchestration.schedule.update_loss_every_n_batch,
    )

    # evaluator
    evaluator = epoch.MultiHeadEvaluator(
        # base engine
        engine_runtime=engine_runtime,
        dataloaders=data_loaders,
        dispatcher=context.dispatcher,
        device=context.device,
        # evaluator-specific
        val_every=config.orchestration.schedule.val_every_n_epoch,
        infer_every=config.orchestration.schedule.infer_every_n_epoch,
        dataset=eval_dataset,
    )

    # return engine with matched mode
    match mode:
        case 'train_eval':
            return epoch.EpochEngine(mode, trainer, evaluator)
        case 'train_only':
            return epoch.EpochEngine(mode, trainer, None)
        case 'eval_only':
            return epoch.EpochEngine(mode, None, evaluator)
