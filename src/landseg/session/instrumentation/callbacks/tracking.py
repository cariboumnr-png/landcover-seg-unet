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

'''Tracking callback'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.instrumentation.callbacks as callbacks
import landseg.session.instrumentation.tracking as tracking

class TrackingCallback(callbacks.BaseCallback):
    '''Tracking callback.'''

    def __init__(
        self,
        trackers: list[typing.Literal['tb', 'mlflow']],
        uri: str,
        # artifact_path: str | None = None,
        **kwargs
    ):
        '''Initialize the callback.'''

        super().__init__(**kwargs)
        self._trackers: list[tracking.BaseTracker] = []
        if 'tb' in trackers:
            self._trackers.append(tracking.TensorBoardTracker(uri))

    def on_train_phase_begin(self, phase: common.PhaseLike): ...

    def on_batch_begin(self, action: str, bidx: int): ...

    def on_train_batch_end(self, bidx: int, results: core.TrainerEpochResults):
        if not results.metrics_updated:
            return
        step = results.global_step
        for tracker in self._trackers:
            tracker.log_scalar('total_loss', results.total_loss, step)
            tracker.log_scalar('lr', results.current_lr or 0.0, step)
            tracker.flush()

    def on_train_step_end(self, results: core.TrainingSessionStep):
        step = results.epoch_in_phase
        for tracker in self._trackers:
            tracker.log_scalar('mean_IoU', results.metrics.target_metrics, step)

    def on_train_phase_end(self, phase: str, reason: str): ...

    def on_train_end(self) -> None:
        for tracker in self._trackers:
            tracker.close()

    def on_checkpointing(self, fp: str): ...
