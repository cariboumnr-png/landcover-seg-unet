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

'''Callback dispatcher for broadcasting session lifecycle events.'''

# local imports
import landseg.core as core
import landseg.session.contracts as contracts
import landseg.session.instrumentation.callbacks.base as base


# ----- public classes
class CallbackDispatcher(contracts.SessionObserverLike):
    '''Broadcast engine events to all registered passive callbacks.'''

    def __init__(self, cbs: list[base.BaseCallback] | None = None):
        '''
        Initialize the callback dispatcher.

        Args:
            cbs:
                Optional initial list of callbacks to register.
        '''
        self.callbacks = cbs or []

    def register(self, callback: base.BaseCallback):
        '''
        Attach a new callback dynamically.

        Args:
            callback:
                Callback instance to register if not already present.
        '''
        if not callback in self.callbacks:
            self.callbacks.append(callback)

    def deregister(self, callback: base.BaseCallback):
        '''
        Remove a registered callback.

        Args:
            callback:
                Callback instance to remove if currently registered.
        '''
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    # megaphone methods
    # session phase begins
    def on_session_phase_begin(self, phase: contracts.PhaseLike) -> None:
        '''
        Broadcast session phase start event to registered callbacks.

        Args:
            phase:
                Phase configuration entering execution.
        '''
        for cb in self.callbacks:
            cb.on_session_phase_begin(phase)

    # session step begins
    def on_session_step_begin(self) -> None:
        '''Broadcast session step start to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_session_step_begin()

    # epoch begins
    def on_epoch_begin(self, epoch: int) -> None:
        '''
        Broadcast epoch start event to registered callbacks.

        Args:
            epoch:
                One-based index of the epoch starting execution.
        '''
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    # policy begins
    def on_train_policy_begin(self) -> None:
        '''Broadcast training policy start to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_train_policy_begin()

    def on_val_policy_begin(self) -> None:
        '''Broadcast validation policy start to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_val_policy_begin()

    def on_infer_policy_begin(self) -> None:
        '''Broadcast inference policy start to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_infer_policy_begin()

    # batch begins
    def on_batch_begin(self, action: str, bidx: int) -> None:
        '''
        Broadcast batch start event to registered callbacks.

        Args:
            action:
                Execution mode name for the batch.
            bidx:
                Zero-based index of the batch starting execution.
        '''
        for cb in self.callbacks:
            cb.on_batch_begin(action, bidx)

    # batch ends
    def on_train_batch_end(
        self, bidx: int, results: core.TrainStepResults
    ) -> None:
        '''
        Broadcast training batch completion to registered callbacks.

        Args:
            bidx:
                Zero-based index of the completed training batch.
            results:
                Training step results for the completed batch.
        '''
        for cb in self.callbacks:
            cb.on_train_batch_end(bidx, results)

    def on_val_batch_end(self) -> None:
        '''Broadcast validation batch end to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_val_batch_end()

    def on_infer_batch_end(self) -> None:
        '''Broadcast inference batch end to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_infer_batch_end()

    # policy ends
    def on_train_policy_end(self, results: core.TrainStepResults) -> None:
        '''
        Broadcast training policy completion to registered callbacks.

        Args:
            results:
                Aggregated training results for the epoch.
        '''
        for cb in self.callbacks:
            cb.on_train_policy_end(results)

    def on_val_policy_end(self, results: core.ValStepResults) -> None:
        '''
        Broadcast validation policy completion to registered callbacks.

        Args:
            results:
                Aggregated validation results for the epoch.
        '''
        for cb in self.callbacks:
            cb.on_val_policy_end(results)

    def on_infer_policy_end(self, results: core.InferStepResults) -> None:
        '''
        Broadcast inference policy completion to registered callbacks.

        Args:
            results:
                Aggregated inference results for the epoch.
        '''
        for cb in self.callbacks:
            cb.on_infer_policy_end(results)

    # epoch ends
    def on_epoch_end(self, epoch: int) -> None:
        '''
        Broadcast epoch completion event to registered callbacks.

        Args:
            epoch:
                One-based index of the completed epoch.
        '''
        for cb in self.callbacks:
            cb.on_epoch_end(epoch)

    # session step ends
    def on_session_step_end(self, results: core.SessionStepSummary) -> None:
        '''
        Broadcast session step completion to registered callbacks.

        Args:
            results:
                Summary of metrics and status for the completed step.
        '''
        for cb in self.callbacks:
            cb.on_session_step_end(results)

    # session phase ends
    def on_session_phase_end(self, phase: str, reason: str) -> None:
        '''
        Broadcast session phase completion to registered callbacks.

        Args:
            phase:
                Name of the completed phase.
            reason:
                Reason the phase terminated.
        '''
        for cb in self.callbacks:
            cb.on_session_phase_end(phase, reason)

    # session end
    def on_session_end(self) -> None:
        '''Broadcast session termination to registered callbacks.'''
        for cb in self.callbacks:
            cb.on_session_end()

    # utilities
    def on_checkpointing(self, fp: str) -> None:
        '''
        Broadcast checkpoint persistence to registered callbacks.

        Args:
            fp:
                File path where the checkpoint was saved.
        '''
        for cb in self.callbacks:
            cb.on_checkpointing(fp)
