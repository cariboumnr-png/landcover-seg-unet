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
Phase-level orchestration policy.

This module defines orchestration logic for executing a training phase,
which consists of multiple epochs. It manages model head configuration,
metric tracking, checkpoint signaling, and optional early stopping.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.orchestration.phases as phases
import landseg.session.orchestration.policy as policy
import landseg.session.orchestration.events as events

@dataclasses.dataclass
class TrackingConfig:
    '''Configuration for metric tracking and early stopping.'''
    track_mode: str = 'max'
    enable_early_stop: bool = False
    patience_epochs: int | None = 5
    delta: float | None = 0.0005

@dataclasses.dataclass
class _MetricsTracker:
    '''
    Internal state container for tracking metric progression.

    This class maintains state across epochs for determining improvements,
    best values, and early stopping conditions.
    '''
    last_value: float = -float('inf')
    curr_value: float = -float('inf')
    best_value: float = -float('inf')
    best_epoch: int = -1
    patience_n: int = 0
    is_best_epoch: bool = False

class PhasePolicy:
    '''
    Orchestrates execution of a training phase across multiple epochs.

    This class coordinates:
    - Model head configuration for the phase
    - Iterative execution of epochs via `EpochPolicy`
    - Metric tracking and best model selection
    - Checkpoint signaling
    - Optional early stopping based on tracked metrics

    It supports both event-driven execution (via generators) and direct
    execution.
    '''

    def __init__(
        self,
        *,
        epoch_runner: common.EpochEngineLike,
        phase_config: phases.PhaseLike,
        track_config: TrackingConfig,
    ):
        '''
        Initializes the phase policy.

        Args:
            epoch_runner: Engine responsible for executing individual
                epochs.
            phase_config: Configuration describing the phase, including
                number of epochs and head settings.
            track_config: Configuration for tracking metrics and early
                stopping behavior.
            start_epoch: Epoch index to start from. Defaults to 1.
        '''

        self.runner = epoch_runner
        self.config = phase_config
        self.track = track_config
        #
        self.tracker = _MetricsTracker()

    @property
    def patience_reached(self) -> bool | None:
        '''Convenience flag; return `True` if patience is reached.'''
        if self.track.patience_epochs is None:
            return None
        return self.tracker.patience_n >= self.track.patience_epochs

    def run(self) -> typing.Generator[events.Event, None, float]:
        '''
        Runs the phase with event emission.

        This method:
        - Configures model heads for the phase
        - Iterates through epochs using `EpochPolicy`
        - Tracks metrics and determines best epochs
        - Emits checkpoint and early stopping events

        Yields:
            Lifecycle events:
                - `PhaseStart` and `PhaseEnd`
                - Epoch-level events (delegated)
                - `MetricsReport` after each epoch
                - `CheckpointRequest` after each epoch
                - `StopRun` if early stopping is triggered

        Returns:
            Best metric value observed during the phase.
        '''

        # set trainer head state per phase
        self.runner.set_head_state(
            self.config.active_heads,
            self.config.frozen_heads,
        )
        # TBD set learning rate per phase

        # phase starts
        yield events.PhaseStart(self.config.name)

        # iterate epochs
        for epoch in range(self.config.start_epoch, self.config.num_epochs + 1):

            # delegate to epoch policy
            metrics = yield from policy.EpochPolicy(
                epoch_runner=self.runner,
                phase_name=self.config.name,
                epoch_index=epoch,
                active_heads=self.config.active_heads
            ).run()

            # track metrics
            # - if validation is run this epoch
            if metrics.validation:
                tracked = self._track(epoch, metrics.validation.target_metrics)
            # - if validation is not run this epoch
            else:
                tracked = (0.0, -1, False)

            # report tracking results
            yield events.MetricsReport(*tracked, metrics)

            # request checkpointing
            tag = 'best' if self.tracker.is_best_epoch else 'last'
            yield events.CheckpointRequest(tag)

            # early stop check
            if not self.track.enable_early_stop:
                continue
            if self.patience_reached:
                yield events.StopRun('Patience limit reached')
                break

        # reset trainer head state
        self.runner.reset_head_state()

        # phase ends
        yield events.PhaseEnd(self.config.name)

        # Return the best value tracked during this phase
        return self.tracker.best_value

    def execute(self) -> list[core.EpochResults]:
        '''
        Executes the phase without emitting events.

        This method execute all epochs sequentially and collects their
        metrics without any orchestration events.

        Returns:
            List of metrics for each executed epoch.
        '''

        epochs: list[core.EpochResults] = []
        for epoch in range(1, self.config.num_epochs + 1):
            epoch_metrics = policy.EpochPolicy(
                epoch_runner=self.runner,
                phase_name=self.config.name,
                epoch_index=epoch,
                active_heads=self.config.active_heads
            ).execute()
            epochs.append(epoch_metrics)
        return epochs

    def _track(
        self,
        epoch: int,
        target_metrics: float
    ) -> tuple[float, int, bool]:
        '''Track best metrics and count patience epochs.'''

        # update last and current value
        if epoch == 1:
            self.tracker.last_value = 0.0
            self.tracker.curr_value = target_metrics
        else:
            self.tracker.last_value = self.tracker.curr_value
            self.tracker.curr_value = target_metrics

        # track by mode
        mode = self.track.track_mode
        delta = self.track.delta or 0.0
        match mode:
            # track if metrics is increasing
            case 'max':
                if target_metrics >= self.tracker.best_value + delta:
                    self.tracker.best_value = target_metrics
                    self.tracker.best_epoch = epoch
                    self.tracker.patience_n = 0
                    self.tracker.is_best_epoch = True
                else:
                    self.tracker.is_best_epoch = False
                    self.tracker.patience_n += 1
            case _:
                raise ValueError(f'Invalid track mode: {mode}')

        # return reporting attributes
        return (
            self.tracker.best_value,
            self.tracker.best_epoch,
            self.tracker.is_best_epoch,
        )
