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
Callback dispatcher.
'''

# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.instrumentation.callbacks as callbacks

class CallbackDispatcher(common.SessionObserverLike):
    '''Broadcast engine events to all registered passive callbacks.'''

    def __init__(self, cbs: list[callbacks.BaseCallback] | None = None):
        '''Initialize the dispatcher'''

        self.callbacks = cbs or []

    def register(self, callback: callbacks.BaseCallback):
        '''Attach a new callback dynamically.'''

        if not callback in self.callbacks:
            self.callbacks.append(callback)

    def deregister(self, callback: callbacks.BaseCallback):
        '''Remove a callback.'''

        if callback in self.callbacks:
            self.callbacks.remove(callback)

    # megaphone methods
    # --- training phase begins
    def on_train_phase_begin(self, phase: common.PhaseLike) -> None:
        for cb in self.callbacks:
            cb.on_train_phase_begin(phase)

    # --- training step begins
    def on_train_step_begin(self) -> None: ...

    # --- epoch begins
    def on_epoch_begin(self, epoch: int) -> None: ...

    # --- policy begins
    def on_train_policy_begin(self) -> None: ...

    def on_val_policy_begin(self) -> None: ...

    def on_infer_policy_begin(self) -> None: ...

    # --- batch begins
    def on_batch_begin(self, action: str, bidx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(action, bidx)

    # --- batch ends
    def on_train_batch_end(self, bidx: int, results: core.TrainerEpochResults) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_end(bidx, results)

    def on_val_batch_end(self) -> None: ...

    def on_infer_batch_end(self) -> None: ...

    # --- policy ends
    def on_train_policy_end(self, results: core.TrainerEpochResults) -> None:
        for cb in self.callbacks:
            cb.on_train_policy_end(results)

    def on_val_policy_end(self, results: core.EvaluatorEpochResults) -> None:
        for cb in self.callbacks:
            cb.on_val_policy_end(results)

    def on_infer_policy_end(self, results: core.EvaluatorEpochResults) -> None:
        for cb in self.callbacks:
            cb.on_infer_policy_end(results)

    # --- epoch ends
    def on_epoch_end(self, epoch: int) -> None: ...

    # --- training step ends
    def on_train_step_end(self, results: core.TrainingSessionStep) -> None:
        for cb in self.callbacks:
            cb.on_train_step_end(results)

    # --- training phase ends
    def on_train_phase_end(self) -> None: ...

    # --- utilities
    def on_checkpointing(self, fp: str) -> None:
        for cb in self.callbacks:
            cb.on_checkpointing(fp)
