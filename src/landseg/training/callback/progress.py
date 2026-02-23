# pylint: disable=protected-access
'''Progress increments'''

# local imports
import landseg.training.callback as callback

class ProgressCallback(callback.Callback):
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
        eval_interval = self.config.schedule.eval_interval
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
        eval_interval = self.config.schedule.eval_interval
        # already at max epoch
        if epoch == self.config.schedule.max_epoch:
            return
        # if validation is done, increment after this hook
        if eval_interval is not None and epoch % eval_interval == 0:
            self.state.progress.epoch += 1
