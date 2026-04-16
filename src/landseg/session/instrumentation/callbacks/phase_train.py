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
'''Train phase callback class.'''

# local imports
import landseg.session.components.callback as callback

class TrainCallback(callback.Callback):
    '''Training: parse - forward - compute loss - backward - step.'''

    def on_train_epoch_begin(self, epoch: int) -> None:
        # reset train loss and logs
        self.state.epoch_sum.train_loss = 0.0
        self.state.epoch_sum.train_logs.head_losses.clear()
        self.state.epoch_sum.train_logs.head_losses_str = ''
        self.state.epoch_sum.train_logs.updated = False

    def on_train_batch_begin(self, bidx: int, batch: tuple) -> None:
        # refresh batch context with new input batch (from training data)
        self.state.batch_cxt.refresh(bidx, batch)
        # refresh batch results
        self.state.batch_out.refresh(bidx)

    def on_train_batch_forward(self) -> None: ...

    def on_train_batch_compute_loss(self) -> None: ...

    def on_train_batch_end(self) -> None: ...

    def on_train_backward(self) -> None: ...

    def on_train_before_optimizer_step(self) -> None: ...

    def on_train_optimizer_step(self) -> None: ...

    def on_train_epoch_end(self) -> None: ...
