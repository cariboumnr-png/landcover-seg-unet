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
Base callback implementation for session observation.

Provides `BaseCallback`, an optional-hook base class implementing
`SessionObserverLike` that subclasses can override to react to training,
validation, inference, and checkpointing events.
'''

# pylint: disable=unused-argument

# standard imports
import pathlib
# local imports
import landseg.core as core
import landseg.session.contracts as contracts
import landseg.session.instrumentation.dashboards as dashboards


# ----- public classes
class BaseCallback(contracts.SessionObserverLike):
    '''
    Base class for defining session observation callbacks.

    Subclasses can override any hook method to execute custom logic at
    specific points in the training, validation, or inference lifecycle.
    All methods are optional no-ops by default; override only those
    needed.
    '''

    def __init__(
        self,
        trackers: list[dashboards.BaseTracker] | None = None,
        *,
        verbose: bool = True,
        label_color_map: dict[str, list[int]] | None = None
    ):
        '''
        Initialize the callback instance.

        Args:
            trackers:
                Optional list of dashboard trackers for logging metrics,
                images, and artifacts.
            verbose:
                Whether to enable verbose logging or console output in
                callback implementations.
            label_color_map:
                Optional mapping from class labels to RGB colors for
                visualization callbacks.
        '''
        self._trackers: list[dashboards.BaseTracker] = []
        if trackers:
            self._trackers = trackers
        self.verbose = verbose
        self._label_color_map = label_color_map

    # session phase begins
    def on_session_phase_begin(self, phase: contracts.PhaseLike) -> None:
        '''
        Handle session phase start event.

        Args:
            phase:
                Phase configuration entering execution.
        '''

    # session step begins
    def on_session_step_begin(self) -> None:
        '''Handle session step start event.'''

    # epoch begins
    def on_epoch_begin(self, epoch: int) -> None:
        '''
        Handle epoch start event.

        Args:
            epoch:
                One-based index of the epoch starting execution.
        '''

    # policy begins
    def on_train_policy_begin(self) -> None:
        '''Handle training policy start event.'''

    def on_val_policy_begin(self) -> None:
        '''Handle validation policy start event.'''

    def on_infer_policy_begin(self) -> None:
        '''Handle inference policy start event.'''

    # batch begins
    def on_batch_begin(self, action: str, bidx: int) -> None:
        '''
        Handle batch start event.

        Args:
            action:
                Execution mode name for the batch.
            bidx:
                Zero-based index of the batch starting execution.
        '''

    # batch ends
    def on_train_batch_end(self, bidx: int, results: core.TrainStepResults) -> None:
        '''
        Handle training batch completion event.

        Args:
            bidx:
                Zero-based index of the completed training batch.
            results:
                Training step results for the completed batch.
        '''

    def on_val_batch_end(self) -> None:
        '''Handle validation batch completion event.'''

    def on_infer_batch_end(self) -> None:
        '''Handle inference batch completion event.'''

    # policy ends
    def on_train_policy_end(self, results: core.TrainStepResults) -> None:
        '''
        Handle training policy completion event.

        Args:
            results:
                Aggregated training results for the epoch.
        '''

    def on_val_policy_end(self, results: core.ValStepResults) -> None:
        '''
        Handle validation policy completion event.

        Args:
            results:
                Aggregated validation results for the epoch.
        '''

    def on_infer_policy_end(self, results: core.InferStepResults) -> None:
        '''
        Handle inference policy completion event.

        Args:
            results:
                Aggregated inference results for the epoch.
        '''

    # epoch ends
    def on_epoch_end(self, epoch: int) -> None:
        '''
        Handle epoch completion event.

        Args:
            epoch:
                One-based index of the completed epoch.
        '''

    # session step ends
    def on_session_step_end(self, results: core.SessionStepSummary) -> None:
        '''
        Handle session step completion event.

        Args:
            results:
                Summary of metrics and status for the completed step.
        '''

    # session phase ends
    def on_session_phase_end(self, phase: str, reason: str) -> None:
        '''
        Handle session phase completion event.

        Args:
            phase:
                Name of the completed phase.
            reason:
                Reason the phase terminated.
        '''

    # session end
    def on_session_end(self) -> None:
        '''Handle session termination event.'''

    # utilities
    def on_checkpointing(self, fp: pathlib.Path) -> None:
        '''
        Handle checkpoint persistence event.

        Args:
            fp:
                File path where the checkpoint was saved.
        '''
