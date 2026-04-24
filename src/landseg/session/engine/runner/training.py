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
Epoch-level training execution runner.

This module provides a small, execution-focused wrapper responsible for
running exactly one training epoch with optional evaluation. It bundles
the trainer and evaluator execution paths into a single, stable unit that
can be invoked by higher-level orchestration or policy code.

Design notes
------------
- This module performs *execution only* and does not implement control
  flow, early stopping, phase logic, or event emission.
- It is intentionally agnostic to phases, curricula, and training
  schedules; those concerns live in the orchestration layer.
- The primary purpose of this abstraction is to provide a clean,
  testable execution façade that can later be wrapped by a
  generator-based training engine.

The API surface is intentionally minimal and expected to remain stable
as orchestration evolves.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.session.engine as engine

@dataclasses.dataclass(frozen=True)
class EpochMetrics:
    '''Immutable metrics produced by a single training epoch.'''
    training: dict[str, float]
    validation: dict[str, dict[str, typing.Any]] | None

# --------------------------------Public  Class--------------------------------
class TrainingEpochRunner:
    '''
    Execute exactly one training epoch with optional validation.

    This class encapsulates all batch-level execution for a single epoch,
    including training and optionally evaluation, and returns aggregated
    metrics. It does not manage epoch iteration, stopping criteria, phase
    transitions, or logging.

    Instances of this class are expected to be orchestrated by higher-
    level control logic (e.g. phase policies or a generator-based runner).
    '''

    def __init__(
        self,
        trainer: engine.MultiHeadTrainer,
        evaluator: engine.MultiHeadEvaluator | None,

    ):
        '''
        Initialize an epoch runner.

        Args:
            trainer: Concrete trainer responsible for executing the
                training pass.
            evaluator: Optional evaluator responsible for running
                validation. If None, no evaluation step will be executed.
        '''

        # parse arguments
        self.trainer = trainer
        self.evaluator = evaluator

    @property
    def has_evaluator(self) -> bool:
        '''Return True if validation will be executed for each epoch.'''
        return self.evaluator is not None

    def run(self, epoch: int) -> EpochMetrics:
        '''
        Run a single training epoch and return aggregated metrics.

        Args:
            epoch: Current epoch index, forwarded to the trainer for
            bookkeeping or logging purposes.

        Returns:
            Immutable summary of training (and optional validation)
            results for this epoch.
        '''

        # train the current epoch
        t_logs = self.trainer.train_one_epoch(epoch)
        # validate each epoch if evalutor is present
        if self.evaluator:
            v_logs = self.evaluator.validate()
        else:
            v_logs = None

        # return training and validation logs
        return EpochMetrics(t_logs, v_logs)
