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
Session construction and wiring.

This module provides a factory (`build_session`) that assembles all
runtime objects required to execute a multi-head model workflow,
including:
- component graph (losses, metrics, etc.)
- runtime state (device placement, AMP configuration)
- instrumentation callbacks
- batch execution engine
- evaluator (always instantiated)
- trainer (always instantiated, but may not be used)
- orchestration runner (only in training mode)

Configuration is split into two distinct inputs:

- `config` (static, user-defined):
  Structured configuration originating from external sources (e.g., Hydra
  dataclasses). It defines *what* to build: components, runtime behavior,
  and training phases.

- `context` (dynamic, invocation-time):
  Execution-specific parameters that define *how* the session is run in a
  given invocation. This includes intent (training/evaluation/overfit),
  device selection, logging, and runtime flags.

This separation allows the same configuration to be reused across
different execution modes without modification, while keeping runtime
concerns explicit and localized.

The entry point is `build_session`, which returns a `SessionExcutables`
container with the constructed evaluator and optional training components
depending on the selected intent.

Type behavior:
    The session factory uses Literal-based typing and overloads to
    encode execution intent at the type level. This allows static type
    checkers to infer the exact return type of `build_session` based on
    the provided context.
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
    Structural typing interface for session configuration.

    Defines the minimum required configuration attributes used to build
    a session.

    Attributes:
        phase_schema: Identifier describing how training phases are
            interpreted by the orchestration layer.
        components:
            Configuration used to construct session components such as
            losses, metrics, and dataloaders.
        runtime:
            Runtime configuration controlling precision, optimization,
            scheduling, and monitoring behavior.
        training_phases:
            Ordered sequence of phase definitions used by the training
            orchestration runner.
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
    '''Epoch engine building context.'''
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
    '''doc'''

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
