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

'''Console printing callback.'''

# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.instrumentation.callbacks as callbacks

class ConsoleCallback(callbacks.BaseCallback):
    '''Progress tracker.'''

    def on_train_phase_begin(self, phase: common.PhaseLike) -> None:
        if self.verbose:
            print('__Phase details__')
            text = '\n'.join([
                f'- Phase Name:\t{phase.name}',
                f'- Max Epochs:\t{phase.num_epochs}',
                f'- LR Scale:\t{phase.lr_scale}',
                f'- Active Heads:\t{phase.active_heads}',
                f'- Frozen Heads:\t{phase.frozen_heads}',
            ])
            print(text)

    def on_batch_begin(self, action: str, bidx: int) -> None:
        if self.verbose:
            print(f'{action}... batch_{bidx:04d}', end='\r', flush=True)

    def on_train_batch_end(self, bidx: int, results: core.TrainerEpochResults) -> None:
        if self.verbose and bidx > 1 and bidx == results.last_updated:
            text_list: list[str] = []
            text_list.append(f'total_loss: {results.total_loss:.4f}')
            text_list.extend([
                f'{h}_loss: {l:.4f}'
                for h, l in results.head_losses.items() if l > 0
            ])
            text_list.append(f'LR: {results.current_lr:.4e}')
            print(f'batch_{bidx:04d} | ' + '|'.join(text_list))

    def on_train_step_end(self, results: core.TrainingSessionStep) -> None:
        metrics = results.metrics
        if self.verbose:
             # training metrics is always neends
            assert metrics.training
            # validation may or may not be run every epoch
            if metrics.evaluation:
                mean_iou = metrics.evaluation.target_metrics
            else:
                mean_iou = 0.0
            # best so far
            msg = (
                f'|Total Loss: {metrics.training.total_loss:.4f}|'
                f'Mean IoU: {mean_iou:.4f}|'
                f'Best Epoch: {results.best_epoch_so_far}|'
                f'Best Value: {results.best_value_so_far:.4f}|'
            )
            t = results.phase_max_epoch
            n = len(str(t))
            print(f'[Epoch {results.epoch_in_phase:0{n}d}/{t}] {msg}')

    def on_checkpointing(self, fp: str) -> None:
        print(f'Checkpoint saved: {fp}')
