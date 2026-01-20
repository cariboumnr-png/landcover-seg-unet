'''Progress increments'''

# local imports
import training.callback

class ProgressCallback(training.callback.Callback):
    '''Progress tracker.'''

    def on_train_epoch_begin(self, epoch: int) -> None:
        self.state.progress.epoch = epoch   # get current epoch
        self.state.progress.epoch_step = 0  # reset epoch step

    def on_train_batch_end(self) -> None:
        self.state.progress.epoch_step += 1
        self.state.progress.global_step += 1

    def on_train_epoch_end(self) -> None:
        if self.config.schedule.eval_interval is None or \
            self.state.progress.epoch % self.config.schedule.eval_interval != 0:
            self.state.progress.epoch += 1

    def on_validation_begin(self) -> None: ...

    def on_validation_end(self) -> None:
        if self.config.schedule.eval_interval is not None and \
            self.state.progress.epoch % self.config.schedule.eval_interval == 0:
            self.state.progress.epoch += 1
