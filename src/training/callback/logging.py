# pylint: disable=protected-access
'''Logging callback class.'''

# local imports
import training.callback

class LoggingCallback(training.callback.Callback):
    '''Controlled console logging.'''

    def on_train_epoch_begin(self, epoch: int) -> None:
        self.log_train('INFO', f'Epoch_{epoch:03d} training started')

    def on_train_batch_begin(self, bidx: int, batch: tuple) -> None:
        print(f'Processing batch_{bidx:04d}', end='\r', flush=True)

    def on_train_batch_end(self) -> None:
        # train logs to console if updated
        if self.state.epoch_sum.train_logs.updated:
            msg = self.state.epoch_sum.train_logs.head_losses_str
            self.log_train('INFO', msg)

    def on_train_epoch_end(self) -> None:
        # train logs to consle
        epoch = self.state.progress.epoch
        self.log_train('INFO', self.state.epoch_sum.train_logs.head_losses_str)
        self.log_train('INFO', f'Epoch_{epoch:03d} training finished')

    def on_validation_begin(self) -> None:
        epoch = self.state.progress.epoch
        self.log_valdn('INFO', f'Epoch_{epoch:03d} validating started')

    def on_validation_batch_begin(self, bidx: int, batch: tuple) -> None:
        print(f'Processing batch_{bidx:04d}', end='\r', flush=True)

    def on_validation_end(self) -> None:
        epoch = self.state.progress.epoch
        target_head = self.config.monitor.head
        v = self.trainer.state.metrics.best_value
        e = self.trainer.state.metrics.best_epoch
        for t in self.state.epoch_sum.val_logs.head_metrics_str[target_head]:
            self.log_valdn('INFO', t)
        self.log_valdn('INFO', f'Current best metric/epoch:\t{v:.4f}|{e}')
        self.log_valdn('INFO', f'Epoch_{epoch:03d} validation finished')
