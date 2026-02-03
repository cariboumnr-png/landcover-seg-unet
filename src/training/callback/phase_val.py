# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
'''Batch pipeline callbacks.'''

# local imports
import training.callback

class ValCallback(training.callback.Callback):
    '''
    Validation pipeline: parse - forward - compute metrics.
    '''

    def on_validation_begin(self) -> None:
        # set model to validation mode
        self.trainer.comps.model.eval()
        # reset per-head confusion matrix from active heads
        assert self.trainer.state.heads.active_hmetrics is not None
        for metrics_mod in self.trainer.state.heads.active_hmetrics.values():
            metrics_mod.reset(self.trainer.device)
        # reset validation loss and logs
        self.state.epoch_sum.val_loss = 0.0 # currently not in use
        self.state.epoch_sum.val_logs.head_metrics.clear()
        self.state.epoch_sum.val_logs.head_metrics_str.clear()

    def on_validation_batch_begin(self, bidx: int, batch: tuple) -> None:
        # refresh batch context with new input batch (from validation data)
        self.state.batch_cxt.refresh(bidx, batch)
        # refresh batch results
        self.state.batch_out.refresh(bidx)
        # parse from the new batch
        self.trainer._parse_batch()

    def on_validation_batch_forward(self) -> None:
        # get x and domain (optional)
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with autocast context
        with self.trainer._val_ctx():
            outputs = self.trainer.comps.model.forward(x, **domain)
        self.state.batch_out.preds = dict(outputs)

    def on_validation_batch_end(self) -> None:
        # update per-head confusion matrix
        self.trainer._update_conf_matrix()

    def on_validation_end(self) -> None:
        # compute iou
        self.trainer._compute_iou()
        # update experimental level metrics
        self.trainer._track_metrics()
