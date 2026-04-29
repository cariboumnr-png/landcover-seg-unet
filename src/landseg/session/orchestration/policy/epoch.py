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
Epoch-level orchestration policy.

This module defines a lightweight orchestration layer responsible for
managing the lifecycle of a single training epoch. It emits structured
events around epoch execution and delegates the actual computation to a
training engine.
'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.orchestration.events as events

class EpochPolicy:
    '''
    Orchestrates the execution of a single training epoch.

    This class wraps a `TrainingEpochRunner` and is responsible for:
    - Emitting lifecycle events (start and end of an epoch)
    - Delegating execution to the underlying training engine
    - Returning computed epoch metrics

    It supports both generator-based execution (for event streaming) and
    direct execution (for simplified usage).
    '''

    def __init__(
        self,
        *,
        epoch_runner: common.EpochEngineLike,
        phase_name: str,
        epoch_index: int,
        active_heads: list[str] | None = None
    ):
        '''
        Initializes the epoch policy.

        Args:
            training_engine: Engine responsible for executing the epoch
                logic.
            phase_name: Name of the phase (e.g., 'train', 'validation').
            epoch_index: Index of the epoch to run.
            active_heads: Subset of model heads to activate during this
                epoch. Defaults to None.
        '''

        self.epoch = epoch_index
        self.phase = phase_name
        self.runner = epoch_runner
        self.active_heads = active_heads

    def run(self) -> typing.Generator[events.Event, None, core.EpochResults]:
        '''
        Runs the epoch with event emission.

        This method emits structured events before and after delegating
        execution to the underlying training engine. It is designed to be
        used in event-driven pipelines via ``yield from``.

        Yields:
            Lifecycle events:
                - ``EpochStart`` before execution
                - ``EpochEnd`` after execution, including metrics

        Returns:
            Metrics produced by the training engine for this epoch.
        '''
        # epoch starts
        yield events.EpochStart(self.epoch, self.phase)

        # delegate execution to epoch runner
        epoch_metrics = self.runner.run_epoch(self.epoch)

        # epoch ends
        yield events.EpochEnd(self.epoch, self.phase)

        # enables downstream `yield from`
        return epoch_metrics

    def execute(self) -> core.EpochResults:
        '''
        Executes the epoch without emitting events.

        This is a simplified execution path that directly calls the
        underlying training engine without producing any orchestration
        events.

        Returns:
            engine.EpochMetrics: Metrics produced by the training engine.
        '''
        return self.runner.run_epoch(self.epoch)
