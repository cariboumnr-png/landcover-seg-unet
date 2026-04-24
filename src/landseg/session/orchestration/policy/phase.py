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
Phase level orchestration policy.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.session.engine as engine
import landseg.session.orchestration.policy as policy
import landseg.session.orchestration.event as events

@dataclasses.dataclass
class TrackingConfig:
    '''doc'''
    enable_early_stop: bool = False
    patience_epochs: int = 5
    track_mode: str = 'max'
    delta: float = 0.0005

@dataclasses.dataclass
class PhaseConfig:
    '''doc'''
    phase_name: str
    max_epoch: int
    active_heads: list[str]
    frozen_heads: list[str] | None

@dataclasses.dataclass
class _MetricsTracker:
    '''doc'''
    last_value: float = -float('inf')
    curr_value: float = -float('inf')
    best_value: float = -float('inf')
    best_epoch: int = -1
    patience_n: int = 0
    is_best_epoch: bool = False

class PhasePolicy:
    '''doc'''

    def __init__(
        self,
        *,
        training_engine: engine.TrainingEpochRunner,
        phase_config: PhaseConfig,
        track_config: TrackingConfig
    ):
        '''doc'''

        self.engine = training_engine
        self.config = phase_config
        self.track = track_config
        #
        self.tracker = _MetricsTracker()

    def run(self) -> typing.Iterator[events.Event]:
        '''doc'''

        # set trainer head state per phase
        self.engine.trainer.set_head_state(
            self.config.active_heads,
            self.config.frozen_heads,
        )

        # phase starts
        yield events.PhaseStart(self.config.phase_name)

        # iterate epochs
        for epoch in range(1, self.config.max_epoch + 1):

            # delegate to epoch policy
            metrics = yield from policy.EpochPolicy(
                training_engine=self.engine,
                phase_name=self.config.phase_name,
                epoch_index=epoch,
                active_heads=self.config.active_heads
            ).run()

            # track metrics
            self._track_metrics(epoch, metrics)

            # request checkpointing
            if self.tracker.is_best_epoch:
                tag = f'phase_{self.config.phase_name}_epoch_{epoch}_best'
            else:
                tag = f'phase_{self.config.phase_name}_epoch_{epoch}_last'
            yield events.CheckpointRequest(tag)

            # early stop check
            if not self.track.enable_early_stop:
                continue
            if self.tracker.patience_n >= self.track.patience_epochs:
                yield events.StopRun('Patience limit reached')
                break

        # reset trainer head state
        self.engine.trainer.reset_head_state()

        # phase ends
        yield events.PhaseEnd(self.config.phase_name)

    def execute(self):
        '''Run the underlying epoch policy and return raw metrics.'''

        for epoch in range(1, self.config.max_epoch + 1):
            yield policy.EpochPolicy(
                training_engine=self.engine,
                phase_name=self.config.phase_name,
                epoch_index=epoch,
                active_heads=self.config.active_heads
            ).execute()

    def _track_metrics(
        self,
        epoch: int,
        metrics: dict[str, float]
    ):
        '''Track best metrics and count patience epochs.'''

        # retrieve iou metrics
        mean = metrics.get('mean_iou_active_heads', 0.0) # all classes
        mean_ac = metrics.get('mean_ac_iou_active_heads', 0.0) # active classes
        # pick value
        if not any([mean, mean_ac]):
            raise ValueError(f'No valid metrics: {metrics}')
        target = mean_ac if mean_ac else mean

        # update last and current value
        if epoch == 1:
            self.tracker.last_value = 0.0
            self.tracker.curr_value = target
        else:
            self.tracker.last_value = self.tracker.curr_value
            self.tracker.curr_value = target

        # track by mode
        mode = self.track.track_mode
        match mode:
            # track if metrics is increasing
            case 'max':
                if target >= self.tracker.best_value + self.track.delta:
                    self.tracker.best_value = target
                    self.tracker.best_epoch = epoch
                    self.tracker.patience_n = 0
                    self.tracker.is_best_epoch = True
                else:
                    self.tracker.patience_n += 1
            case _:
                raise ValueError(f'Invalid track mode: {mode}')
