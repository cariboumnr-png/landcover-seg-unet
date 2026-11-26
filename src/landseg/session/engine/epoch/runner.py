# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Epoch-level execution runner.

Provides a minimal execution wrapper responsible for running a single
epoch of training and/or evaluation. This module isolates batch-level
execution from higher-level orchestration concerns.

Design principles:
    - Execution-only: no control flow, scheduling, or early stopping.
    - Orchestration-agnostic: no phases, curricula, or policies.
    - Stable interface: thin, testable facade composable by external
      orchestration layers (e.g., generator-based engines).
'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.engine.epoch.policy as policy


# ----- typing aliases
Mode: typing.TypeAlias = typing.Literal['train_eval', 'train_only', 'eval_only']


# ----- public classes
class EpochRunner:
    '''
    Execute exactly one training epoch with optional validation.

    This class encapsulates batch-level execution for a single epoch,
    including training and optionally evaluation, and returns aggregated
    metrics. It does not manage epoch loops, stopping criteria, phase
    transitions, or logging.

    Instances of this class are expected to be orchestrated by higher-
    level control logic (e.g. phase policies or a generator runner).
    '''

    @typing.overload
    def __init__(self,
        mode: typing.Literal['train_eval'],
        trainer: policy.MultiHeadTrainer,
        evaluator: policy.MultiHeadEvaluator
    ) -> None: ...

    @typing.overload
    def __init__(self,
        mode: typing.Literal['train_only'],
        trainer: policy.MultiHeadTrainer,
        evaluator: None
    ) -> None: ...

    @typing.overload
    def __init__(self,
        mode: typing.Literal['eval_only'],
        trainer: None,
        evaluator: policy.MultiHeadEvaluator
    ) -> None: ...

    def __init__(
        self,
        mode: typing.Literal['train_eval', 'train_only', 'eval_only'],
        trainer: policy.MultiHeadTrainer | None,
        evaluator: policy.MultiHeadEvaluator | None,
    ):
        '''
        Initialize the epoch runner with execution mode and policies.

        Args:
            mode:
                Execution mode (`'train_eval'`, `'train_only'`, or
                `'eval_only'`).
            trainer:
                Training policy controller, or `None` for `'eval_only'`.
            evaluator:
                Evaluation policy controller, or `None` when
                `'train_only'`.
        '''
        # parse arguments
        self.mode: Mode = mode
        self.trainer: policy.MultiHeadTrainer | None = trainer
        self.evaluator: policy.MultiHeadEvaluator | None = evaluator

    @property
    def training_batch_count(self) -> int:
        '''Return total number of training samples of the epoch.'''
        if not self.trainer:
            return 0
        return self.trainer.dataloaders.meta.patch_count.train

    def run_epoch(self, epoch: int) -> core.SessionStepResults:
        '''
        Run a single training epoch and return aggregated metrics.

        Args:
            epoch:
                Current epoch index, forwarded to trainer and evaluator.

        Returns:
            core.SessionStepResults:
                Training, validation, and inference results.

        Raises:
            ValueError:
                If a required trainer or evaluator policy is missing for
                the configured `mode`.
        '''
        # run by mode
        match self.mode:
            case 'train_eval':
                if not (self.trainer and self.evaluator):
                    raise ValueError('Missing trainer or evaluator')
                _train = self.trainer.train_one_epoch(epoch)
                _val = self.evaluator.validate(epoch)
                _infer = self.evaluator.infer(epoch)
                return core.SessionStepResults(_train, _val, _infer)

            case 'train_only':
                if not self.trainer:
                    raise ValueError('Missing trainer')
                _train = self.trainer.train_one_epoch(epoch)
                return core.SessionStepResults(_train, None, None)

            case 'eval_only':
                if not self.evaluator:
                    raise ValueError('Missing evaluator')
                _val = self.evaluator.validate(0) # will always run
                return core.SessionStepResults(None, _val, None)

    def set_head_state(
        self,
        active_heads: list[str] | None = None,
        frozen_heads: list[str] | None = None,
    ) -> None:
        '''
        Set head state for trainer or evaluator when present.

        Args:
            active_heads:
                Head names to activate, or `None` for all heads.
            frozen_heads:
                Head names whose parameters remain frozen.
        '''
        if self.trainer:
            self.trainer.set_head_state(active_heads, frozen_heads)
        if self.evaluator:
            self.evaluator.set_head_state(active_heads, frozen_heads)

    def reset_head_state(self) -> None:
        '''Reset head state for trainer or evaluator when present.'''
        if self.trainer:
            self.trainer.reset_head_state()
        if self.evaluator:
            self.evaluator.reset_head_state()
