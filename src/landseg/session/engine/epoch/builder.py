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
Epoch runner construction utilities.

Constructs training and evaluation policies over an execution runtime
and binds them into an `EpochRunner`.

Public APIs:
    - `ScheduleConfigShape`: protocol for schedule frequencies.
    - `build_epoch_runner`: construct configured epoch runner.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.session.contracts as contracts
import landseg.session.engine.epoch.policy as policy
import landseg.session.engine.epoch.runner as runner
import landseg.session.engine.protocols as protocols


# ----- public types
class ScheduleConfigShape(typing.Protocol):
    '''Configuration interface for epoch execution frequencies.'''
    @property
    def update_loss_every_n_batch(self) -> int: ...
    @property
    def val_every_n_epoch(self) -> int: ...
    @property
    def infer_every_n_epoch(self) -> int: ...


# ----- public dataclasses
@dataclasses.dataclass
class EpochRunnerContext:
    '''Externally required context for building the epoch runner.'''
    runtime: policy.EngineRuntime
    dataloaders: protocols.DataLoadersLike
    dispatcher: contracts.SessionObserverLike


# ----- public functions
def build_epoch_runner(
    context: EpochRunnerContext,
    schedule: ScheduleConfigShape,
    *,
    mode: runner.Mode = 'train_eval',
    eval_dataset: typing.Literal['val', 'test'] = 'val',
    device: str | None = None,
) -> runner.EpochRunner:
    '''
    Construct an epoch runner with training and/or evaluation policies.

    Assembles `MultiHeadTrainer` and/or `MultiHeadEvaluator` policies
    bound to the provided runtime execution core and returns an
    `EpochRunner` configured for the specified mode.

    Args:
        engine_runtime:
            Runtime container holding batch executor, optimizer, and
            tasks.
        dataloaders:
            Dataset loaders providing training and evaluation batches.
        dispatcher:
            Event dispatcher for session lifecycle callbacks.
        mode:
            Execution mode ('train_eval', 'train_only', 'eval_only').
        schedule:
            Optional schedule configuration specifying execution
            frequencies.
        device:
            Target execution device. If omitted, defaults to the
            device configured on the underlying batch engine.
        eval_dataset:
            Dataset split used for inference ('val' or 'test').

    Returns:
        runner_mod.EpochRunner:
            Configured epoch runner ready for epoch-wise execution.
    '''
    target_device = (
        device if device is not None else context.runtime.engine.device
    )

    trainer: policy.MultiHeadTrainer | None = None
    evaluator: policy.MultiHeadEvaluator | None = None

    if mode in {'train_eval', 'train_only'}:
        trainer = policy.MultiHeadTrainer(
            engine_runtime=context.runtime,
            dataloaders=context.dataloaders,
            dispatcher=context.dispatcher,
            device=target_device,
            update_every=schedule.update_loss_every_n_batch,
        )

    if mode in {'train_eval', 'eval_only'}:
        evaluator = policy.MultiHeadEvaluator(
            engine_runtime=context.runtime,
            dataloaders=context.dataloaders,
            dispatcher=context.dispatcher,
            device=target_device,
            val_every=schedule.val_every_n_epoch,
            infer_every=schedule.infer_every_n_epoch,
            dataset=eval_dataset,
        )

    match mode:
        case 'train_eval':
            assert trainer is not None and evaluator is not None
            return runner.EpochRunner(mode, trainer, evaluator)
        case 'train_only':
            assert trainer is not None
            return runner.EpochRunner(mode, trainer, None)
        case 'eval_only':
            assert evaluator is not None
            return runner.EpochRunner(mode, None, evaluator)
