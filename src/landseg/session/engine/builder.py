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
import landseg.session.contracts as contracts
import landseg.session.engine.batch as batch
import landseg.session.engine.epoch as epoch
import landseg.session.engine.optim as optim
import landseg.session.engine.protocols as protocols
import landseg.session.engine.tasks as tasks


# ----- private types
class _EpochEngineConfigShape(typing.Protocol):
    '''
    Configuration interface for constructing the epoch engine.

    Defines the required configuration sections used to build data
    loaders, execution runtime, optimization, task components, and
    orchestration scheduling behavior.
    '''
    @property
    def engine_exec(self) -> batch.BatchExecConfigShape: ...
    @property
    def engine_optim(self) -> optim.OptimConfigShape: ...
    @property
    def engine_tasks(self) -> tasks.TaskConfigShape: ...


# ----- public dataclasses
@dataclasses.dataclass
class EpochEngineContext:
    '''Runtime context required for building the epoch engine.'''
    dataspecs: core.DataSpecs
    model: core.MultiheadModelLike
    schedule: protocols.ScheduleConfigShape
    dispatcher: contracts.SessionObserverLike
    device: str
    logger: common.SessionLogger | None = None


# ----- public functions
def build_engine(
    dataloaders: protocols.DataLoadersLike,
    context: EpochEngineContext,
    config: _EpochEngineConfigShape,
    *,
    mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
    eval_dataset: typing.Literal['val', 'test'] = 'val'
) -> epoch.EpochRunner:
    '''
    Construct the full execution engine for training and/or evaluation.

    Assembles dataloaders, batch execution engine, optimization wrapper,
    and task components into an execution runtime, then constructs the
    epoch runner configured for the specified mode.
    '''

    # data loader spatial division compability
    p = dataloaders.meta.patch_size
    s = context.model.spatial_divisor
    if not p % s == 0:
        raise ValueError(
            f'Invalid patch dimension: patch size ({p}) is not divisible '
            f'by spatial divisor ({s})'
        )

    # build engine runtime
    batch_engine = batch.build_batch_engine(
        context.dataspecs,
        dataloaders,
        context.model,
        config.engine_exec,
        device=context.device
    )

    optimization = optim.build_optimization(
        context.model,
        config.engine_optim
    )

    engine_tasks = tasks.build_engine_tasks(
        context.dataspecs,
        config.engine_tasks
    )

    engine_runtime = epoch.EngineRuntime(
        engine=batch_engine,
        engine_optim=optimization,
        engine_tasks=engine_tasks,
    )

    return epoch.build_epoch_runner(
        engine_runtime,
        dataloaders,
        context.dispatcher,
        mode=mode,
        schedule=context.schedule,
        device=context.device,
        eval_dataset=eval_dataset,
    )
