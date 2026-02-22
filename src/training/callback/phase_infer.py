# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
'''Inference phase callback class.'''

# local imports
import training.callback

class InferCallback(training.callback.Callback):
    '''
    Inference pipeline: parse -> forward -> collect outputs (optional).
    '''

    def on_inference_begin(self) -> None:
        # model to eval mode
        self.trainer.comps.model.eval()
        # reset/prepare inference accumulators if you aggregate
        # A dict[str, list[Tensor]]: head -> list of batched preds
        self.state.epoch_sum.infer_output.maps.clear()
        # self.state.epoch_sum.infer_meta = []  # optional: stash block ids, etc.
        # You can also reset a text log if you like:
        # self.state.epoch_sum.infer_logs_text = ''

    def on_inference_batch_begin(self, bidx: int, batch: tuple) -> None:
        # refresh batch ctx
        self.state.batch_cxt.refresh(bidx, batch)
        self.state.batch_cxt.block_index_range = (-1, -1)
        # refresh batch results
        self.state.batch_out.refresh(bidx)
        # parse the batch (x, domain), y may be empty [B, 0]
        self.trainer._parse_batch()
        # resolve block index range from this batch
        self.trainer._resolve_batch_block_range()

    def on_inference_batch_forward(self) -> None:
        # get x and (optional) domain
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with inference + autocast (same style as validation)
        with self.trainer._val_ctx():
            outputs = self.trainer.comps.model.forward(x, **domain)
        # store raw predictions to batch output
        self.state.batch_out.preds = dict(outputs)

    def on_inference_batch_end(self) -> None:
        # aggregate to epoch storage (CPU detach)
        self.trainer._aggregate_batch_predictions()

    def on_inference_end(self, out_dir: str) -> None:
        # stitch all blocks together and output a preview
        self.trainer._preview_monitor_head(out_dir)
