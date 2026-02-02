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
        self.state.epoch_sum.infer_outputs.maps.clear()
        # self.state.epoch_sum.infer_meta = []  # optional: stash block ids, etc.
        # You can also reset a text log if you like:
        # self.state.epoch_sum.infer_logs_text = ''

    def on_inference_batch_begin(self, bidx: int, batch: tuple) -> None:
        # refresh batch ctx
        self.state.batch_cxt.refresh(bidx, batch)
        # refresh batch results
        self.state.batch_out.refresh(bidx)
        # parse the batch (x, domain), y may be empty [B, 0]
        self.trainer._parse_batch()

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
        # Optional: also store domain info for bookkeeping
        # (e.g., ids to map predictions back to data units)
        # self.state.epoch_sum.infer_meta.append(self.state.batch_cxt.domain)

    def on_inference_end(self) -> None:
        # Optional: finalize logs, or run any per-epoch post-processing.
        # You could also concatenate lists here if desired:
        # for h, parts in self.state.epoch_sum.infer_preds.items():
        #     self.state.epoch_sum.infer_preds[h] = torch.cat(parts, dim=0)
        pass
