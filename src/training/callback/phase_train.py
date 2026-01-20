# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
'''Batch pipeline callbacks.'''

# local imports
import training.callback

class TrainCallback(training.callback.Callback):
    '''
    Training pipeline: parse - forward - compute loss - backward - step.
    '''

    def on_train_epoch_begin(self, epoch: int) -> None:
        # set model to train mode
        self.trainer.comps.model.train()
        # reset train loss and logs
        self.state.epoch_sum.train_loss = 0.0
        self.state.epoch_sum.train_logs.clear()
        self.state.epoch_sum.train_logs_text = ''

    def on_train_batch_begin(self, bidx: int, batch: tuple) -> None:
        # refresh batch context with new input batch (from training data)
        self.state.batch_cxt.refresh(bidx, batch)
        # refresh batch results
        self.state.batch_out.refresh(bidx)
        # parse from the new batch
        self.trainer._parse_batch()
        # reset optimizer gradient
        self.trainer.comps.optimization.optimizer.zero_grad(set_to_none=True)

    def on_train_batch_forward(self) -> None:
        # get x and domain (optional)
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with autocast context
        with self.trainer._autocast_ctx():
            predictions = self.trainer.comps.model.forward(x, **domain)
        # assign predictions to batch output
        self.state.batch_out.preds = dict(predictions)

    def on_train_batch_compute_loss(self) -> None:
        # compute loss with autocast context
        with self.trainer._autocast_ctx():
            self.trainer._compute_loss()

    def on_train_backward(self) -> None:
        # backward with or without AMP
        loss = self.state.batch_out.total_loss
        if self.config.precision.use_amp:
            self.state.optim.scaler.scale(loss).backward()
        else:
            loss.backward()

    def on_train_before_optimizer_step(self) -> None:
        optimizer = self.trainer.comps.optimization.optimizer
        # unscale if use AMP
        if self.config.precision.use_amp:
            self.state.optim.scaler.unscale_(optimizer)
        self.trainer._clip_grad()

    def on_train_optimizer_step(self) -> None:
        optimizer = self.trainer.comps.optimization.optimizer
        # use AMP
        if self.config.precision.use_amp:
            self.state.optim.scaler.step(optimizer)
            self.state.optim.scaler.update() # update scaler
        # no AMP
        else:
            self.trainer.comps.optimization.optimizer.step()

    def on_train_batch_end(self) -> None:
        # accumulate batch loss to epoch-level total loss
        batch_loss = float(self.state.batch_out.total_loss.detach().item())
        self.state.epoch_sum.train_loss += batch_loss
