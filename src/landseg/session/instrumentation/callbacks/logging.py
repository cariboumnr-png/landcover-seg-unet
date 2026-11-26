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

'''Logging callback.'''

# local imports
import landseg.core as core
import landseg.runtime as runtime
import landseg.session.contracts as contracts
import landseg.session.instrumentation.callbacks.base as base


# ----- public classes
class LoggingCallback(base.BaseCallback):
    '''Logging callback for session lifecycle events.'''

    def __init__(
        self,
        logger: runtime.Logger | None = None,
        *,
        verbose: bool = True
    ):
        '''
        Initialize the logging callback.

        Args:
            logger:
                Optional logger instance for recording session messages.
            verbose:
                Whether verbose console output is enabled.
        '''
        super().__init__(verbose=verbose)
        self.logger = logger

    def on_session_phase_begin(self, phase: contracts.PhaseLike) -> None:
        '''
        Log phase configuration details at phase start.

        Args:
            phase:
                Phase configuration entering execution.
        '''
        if self.logger:
            self.logger.info(f'[START] Phase {phase.name}')
            self.logger.info(f'- Max Epochs:\t{phase.num_epochs}')
            self.logger.info(f'- LR Scale:\t{phase.lr_scale}')
            self.logger.info(f'- Active Heads:\t{phase.active_heads}')
            self.logger.info(f'- Frozen Heads:\t{phase.frozen_heads}')

    def on_batch_begin(self, action: str, bidx: int) -> None:
        '''
        Log debug message at batch start.

        Args:
            action:
                Execution mode name for the batch.
            bidx:
                Zero-based index of the batch starting execution.
        '''
        if self.logger:
            self.logger.debug(f'{action}... batch_{bidx:04d}')

    def on_train_batch_end(self, bidx: int, results: core.TrainStepResults) -> None:
        '''
        Log training batch losses, regularization, and learning rate.

        Args:
            bidx:
                Zero-based index of the completed training batch.
            results:
                Training step results for the completed batch.
        '''
        if self.logger and results.metrics_updated:
            text_list: list[str] = []
            text_list.append(f'total_obj: {results.total_objective:.4f}')
            text_list.extend([
                f'{h}_loss: {l:.4f}'
                for h, l in results.head_losses.items() if l > 0
            ])
            text_list.extend([
                f'{r}_reg: {v:.4f}'
                for r, v in results.regularization.items()
            ])
            text_list.append(f'LR: {results.current_lr:.4e}')
            msg = f'batch_{bidx:04d} | ' + '|'.join(text_list)
            self.logger.debug(msg)

    def on_session_step_end(self, results: core.SessionStepSummary) -> None:
        '''
        Log epoch summary metrics at session step completion.

        Args:
            results:
                Summary of metrics and status for the completed step.
        '''
        metrics = results.raw_metrics
        if self.logger:
            # training metrics is always present
            assert metrics.training
            # best so far
            msg = (
                f'|Total Objective: {metrics.training.total_objective:.4f}'
                f'|Tracking Metrics: {metrics.target_metrics:.4f}'
                f'|Best Epoch: {results.best_epoch_so_far}'
                f'|Best Value: {results.best_value_so_far:.4f}|'
            )
            t = results.phase_max_epoch
            n = len(str(t))
            epoch_str = f'[Epoch {results.epoch_in_phase:0{n}d}/{t}] {msg}'
            self.logger.info(epoch_str)

    def on_session_phase_end(self, phase: str, reason: str) -> None:
        '''
        Log completion banner when a session phase ends.

        Args:
            phase:
                Name of the completed phase.
            reason:
                Reason the phase terminated.
        '''
        if self.logger:
            self.logger.info(f'[COMPLETE] Phase {phase} ({reason})')

    def on_checkpointing(self, fp: str) -> None:
        '''
        Log the file path of a newly saved checkpoint.

        Args:
            fp:
                File path where the checkpoint was saved.
        '''
        if self.logger:
            self.logger.info(f'Checkpoint saved: {fp}')
