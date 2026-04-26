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

'''Progress increments callback class.'''

# local imports
import landseg.session.instrumentation.callbacks as callbacks

class ProgressCallback(callbacks.Callback):
    '''Progress tracker.'''

    def on_train_epoch_begin(self, epoch: int) -> None:
        self.state.progress.epoch = epoch   # get current epoch
        self.state.progress.epoch_step = 0  # reset epoch step
        if self.verbose:
            print(f'Epoch_{epoch:03d} training started')

    def on_train_batch_begin(self, bidx: int, batch: tuple) -> None:
        if self.verbose:
            print(f'Processing batch_{bidx:04d}', end='\r', flush=True)

    def on_train_batch_end(self) -> None:
        self.state.progress.epoch_step += 1
        self.state.progress.global_step += 1
        if self.state.epoch_sum.train_logs.updated and self.verbose:
            print(self.state.epoch_sum.train_logs.head_losses_str)

    def on_train_epoch_end(self) -> None:
        epoch = self.state.progress.epoch
        if self.verbose:
            print(self.state.epoch_sum.train_logs.head_losses_str)
        print(f'Epoch_{epoch:03d} training finished')

    def on_validation_begin(self) -> None:
        epoch = self.state.progress.epoch
        if self.verbose:
            print(f'Epoch_{epoch:03d} validating started')

    def on_validation_batch_begin(self, bidx: int, batch: tuple) -> None:
        if self.verbose:
            print(f'Processing batch_{bidx:04d}', end='\r', flush=True)

    def on_validation_end(self) -> None:
        self.state.progress.epoch += 1 # increment epoch counter
        if self.verbose:
            epoch = self.state.progress.epoch
            target_head = self.config.monitor.track_head_name
            print('Validation metrics:')
            for s in self.state.epoch_sum.val_logs.head_metrics_str[target_head]:
                print(s)
            print(f'Epoch_{epoch:03d} validation finished')
