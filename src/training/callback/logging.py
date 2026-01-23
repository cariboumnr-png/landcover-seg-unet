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
        # update train logs at intervals and log to console if updated
        bidx=self.state.batch_cxt.bidx
        flag = self.trainer._update_train_logs(bidx=bidx, flush=False)
        if flag:
            self.log_train('INFO', self.state.epoch_sum.train_logs_text)

    def on_train_epoch_end(self) -> None:
        # flush train logs at the end (regardless of interval)
        epoch = self.state.progress.epoch
        bidx=self.state.batch_cxt.bidx
        self.trainer._update_train_logs(bidx=bidx, flush=True)
        self.log_train('INFO', self.state.epoch_sum.train_logs_text)
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
        for t in self.state.epoch_sum.val_logs_text[target_head]:
            self.log_valdn('INFO', t)
        self.log_valdn('INFO', f'Current best metric/epoch:\t{v:.4f}|{e}')
        self.log_valdn('INFO', f'Epoch_{epoch:03d} validation finished')
