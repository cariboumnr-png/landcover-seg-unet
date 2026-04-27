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
Epoch-level execution runner.

Provides a minimal execution wrapper responsible for running a single
epoch of training and/or evaluation. This module isolates batch-level
execution from higher-level orchestration concerns.

Design principles
-----------------
- Execution-only: no control flow, scheduling, or early stopping logic.
- Orchestration-agnostic: does not encode phases, curricula, or policies.
- Stable interface: intended as a thin, testable façade that can be
  composed by external orchestration layers (e.g., generator-based
  engines).
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.session.engine.policy as policy

@dataclasses.dataclass(frozen=True)
class EpochMetrics:
    '''
    Immutable container for metrics produced during a single epoch.

    Attributes:
        training: Results returned by the trainer for the epoch, or None
            if no training step was executed.
        validation: Results returned by the evaluator, or None if no
            evaluation step was executed.
    '''
    training: policy.TrainerEpochResults | None
    validation: policy.EvaluatorEpochResults | None

    def __str__(self) -> str:
        return '\n'.join([
            'Training Results:',
            str(self.training),
            'Validation Results:',
            str(self.validation),
        ])

# --------------------------------Public  Class--------------------------------
class EpochRunner:
    '''
    Execute exactly one training epoch with optional validation.

    This class encapsulates all batch-level execution for a single epoch,
    including training and optionally evaluation, and returns aggregated
    metrics. It does not manage epoch iteration, stopping criteria, phase
    transitions, or logging.

    Instances of this class are expected to be orchestrated by higher-
    level control logic (e.g. phase policies or a generator-based runner).
    '''

    @typing.overload
    def __init__(self,
        mode: typing.Literal['train_evaluate'],
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
        mode: typing.Literal['evaluate_only'],
        trainer: None,
        evaluator: policy.MultiHeadEvaluator
    ) -> None: ...

    def __init__(
        self,
        mode: typing.Literal['train_evaluate', 'train_only', 'evaluate_only'],
        trainer: policy.MultiHeadTrainer | None,
        evaluator: policy.MultiHeadEvaluator | None,
    ):
        '''
        Execute a single epoch of training and/or evaluation.

        Encapsulates batch-level execution for one epoch, including
        optional training and validation steps. This class does not
        manage iteration across epochs, stopping conditions, logging, or
        phase transitions.

        Intended usage:
            Instantiated and invoked by higher-level orchestration logic,
            such as training policies or engine controllers.
        '''

        # parse arguments
        self.mode = mode
        self.trainer = trainer
        self.evaluator = evaluator

    def run(self, epoch: int) -> EpochMetrics:
        '''
        Run a single training epoch and return aggregated metrics.

        Args:
            epoch: Current epoch index, forwarded to the trainer for
            bookkeeping or logging purposes.

        Returns:
            EpochMetrics instance containing training and/or validation
            results depending on the configured execution mode.
        '''

        # run by mode
        match self.mode:
            case 'train_evaluate':
                if not (self.trainer and self.evaluator):
                    raise ValueError('Missing trainer or evaluator')
                train_results = self.trainer.train_one_epoch(epoch)
                val_results = self.evaluator.validate()
                return EpochMetrics(train_results, val_results)

            case 'train_only':
                if not self.trainer:
                    raise ValueError('Missing trainer')
                train_results = self.trainer.train_one_epoch(epoch)
                return EpochMetrics(train_results, None)

            case 'evaluate_only':
                if not self.evaluator:
                    raise ValueError('Missing evaluator')
                val_results = self.evaluator.validate()
                return EpochMetrics(None, val_results)

            case _:
                raise ValueError(f'Invalid mode: {self.mode}')

    def set_head_state(
        self,
        active_heads: list[str] | None = None,
        frozen_heads: list[str] | None = None,
    ) -> None:
        '''Set head state for trainer or evaluator when present.'''

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
