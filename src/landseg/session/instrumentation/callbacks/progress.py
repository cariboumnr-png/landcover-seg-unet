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

# pylint: disable=protected-access

'''Progress increments callback class.'''

# local imports
import landseg.session.instrumentation.callbacks as callbacks

class ProgressCallback(callbacks.Callback):
    '''Progress tracker.'''

    def on_train_epoch_begin(self, epoch: int) -> None:
        self.state.progress.epoch = epoch   # get current epoch
        self.state.progress.epoch_step = 0  # reset epoch step

    def on_train_batch_end(self) -> None:
        self.state.progress.epoch_step += 1
        self.state.progress.global_step += 1

    def on_train_epoch_end(self) -> None:
        # increment epoch counter
        epoch = self.state.progress.epoch
        eval_interval = self.config.schedule.val_every
        # already at max epoch
        if epoch == self.config.schedule.max_epoch:
            return
        # if no validation after training, increment after this hook
        if eval_interval is None or epoch % eval_interval != 0:
            self.state.progress.epoch += 1

    def on_validation_begin(self) -> None: ...

    def on_validation_end(self) -> None:
        # increment epoch counter
        epoch = self.state.progress.epoch
        eval_interval = self.config.schedule.val_every
        # already at max epoch
        if epoch == self.config.schedule.max_epoch:
            return
        # if validation is done, increment after this hook
        if eval_interval is not None and epoch % eval_interval == 0:
            self.state.progress.epoch += 1
