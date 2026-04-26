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

'''
Batch-level execution engine for session workflows.

This module defines a pure batch execution component that performs
all mathematical and stateful operations required to execute a single
training, validation, or inference batch.

The batch execution engine is responsible for:
- Parsing input batches into runtime tensors
- Managing precision and inference contexts
- Running model forward passes
- Computing losses and updating metric accumulators
- Aggregating inference outputs
- Mutating shared runtime state deterministically

The batch execution engine deliberately does NOT:
- Own optimizer or scheduler logic
- Perform backward passes or parameter updates
- Control epoch structure or phase transitions
- Decide logging cadence, checkpointing, or early stopping
- Emit or manage callback hooks

It answers the question:
    "What happened mathematically in this batch?"

Higher-level controllers (e.g. Trainer, Evaluator, Runner) answer:
    "How should these batch results be used?"
'''

# standard imports
import contextlib
# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.engine.batch as batch

class BatchExecutionEngine:
    '''
    Pure batch execution engine for session workflows.

    This class encapsulates all batch-level execution mechanics shared
    across training, validation, and inference, including:

    - Parsing input batches into tensors
    - Managing precision and evaluation contexts
    - Running model forward passes
    - Computing losses and updating metric accumulators
    - Aggregating inference outputs
    - Writing execution results into shared runtime state

    The batch execution engine executes exactly one batch at a time
    and is intentionally ignorant of:

    - Epoch structure
    - Optimization policy
    - Scheduler behavior
    - Logging cadence
    - Experiment or run orchestration

    The engine operates on a shared RuntimeState instance that is
    constructed externally (e.g. at CLI or Runner level) and shared
    across Runner, Trainer/Evaluator, and this executor.

    In other words, this class defines *execution correctness*, not
    *training or evaluation policy*.
    '''

    def __init__(
        self,
        *,
        model: core.MultiheadModelLike,
        state: common.StateLike,
        parent_map: dict[str, str | None],
        use_amp: bool,
        device: str,
    ):
        '''
        Initialize the batch execution engine.

        Args:
            model:
                Multi-head model used for forward execution.
            state:
                Shared RuntimeState instance used to store batch outputs,
                metric accumulators, and inference results.
            parent_map:
                Mapping used to resolve hierarchical relationships during
                loss and metric computation as `{head: parent_head}`.
            use_amp:
                Whether automatic mixed precision should be enabled for
                training and validation execution.
            device:
                Device identifier (e.g. 'cpu', 'cuda', 'cuda:0') used for
                all tensor placement and execution.
        '''

        # parse arguments
        self.model = model
        self.state = state
        self.parent_map = parent_map
        self.use_amp = use_amp
        # move model to device
        self.device = device
        self.model.to(self.device)

    # ----- Public Method
    def run_train_batch(self) -> None:
        '''
        Execute one training batch.

        This method performs batch-level execution, including:

        - Parsing inputs and targets into runtime tensors
        - Running the forward pass under training precision context
        - Computing total and per-head losses
        - Writing results into shared runtime state

        This method does NOT:
        - Perform backward passes
        - Zero or step optimizers
        - Clip gradients
        - Update schedulers
        - Log or aggregate epoch statistics

        All training policy decisions are owned by the Trainer, which
        consumes values written into RuntimeState after this method
        completes.
        '''

        # ----- batch start
        # get new batch and parse into x, y_dict and domain
        self._parse_batch()

        # ----- batch forward
        # get x and domain (optional)
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with autocast context
        with self._autocast_ctx():
            predictions = self.model.forward(x, **domain)
        # assign predictions to batch output
        self.state.batch_out.preds = dict(predictions)

        # ----- batch compute loss
        # compute loss with autocast context
        with self._autocast_ctx():
            self._compute_loss()

    def run_validate_batch(self) -> None:
        '''
        Execute one validation batch.

        This method performs batch-level execution only. It produces all
        mathematical results required for validation, including:

        - Parsing inputs and targets into runtime tensors
        - Running the forward pass under inference/context-only mode
        - Updating per-head metric accumulators (e.g. confusion matrices)

        This method does NOT:
        - Finalize or compute epoch-level metrics
        - Log or format validation results
        - Control validation loops or stopping criteria

        Epoch-level metric finalization and reporting are owned by the
        Trainer or Evaluator after all validation batches have completed.
        '''

        # ----- batch start
        # get new batch and parse into x, y_dict and domain
        self._parse_batch()

        # ----- batch forward
        # get x and domain (optional)
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with autocast context
        with self._val_ctx():
            outputs = self.model.forward(x, **domain)
        self.state.batch_out.preds = dict(outputs)

        # ----- batch end
        # update per-head confusion matrix
        self._update_conf_matrix()

    def run_infer_batch(self):
        '''
        Execute one inference batch.

        This method performs batch-level execution only. It produces all
        mathematical results required for inference, including:

        - Parsing inputs into runtime tensors (targets may be absent)
        - Running the forward pass under inference context
        - Aggregating prediction outputs into inference runtime storage

        This method does NOT:
        - Produce visualizations or previews
        - Write files or artifacts
        - Interpret or post-process inference results

        All inference side effects and result consumption are owned by
        higher-level controllers (Evaluator, Runner, or callbacks).
        '''

        # ----- batch start
        # parse the batch (x, domain), y may be empty [B, 0]
        self._parse_batch()

        # ----- batch forward
        # get x and (optional) domain
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with inference + autocast (same style as validation)
        with self._val_ctx():
            outputs = self.model.forward(x, **domain)
        # store raw predictions to batch output
        self.state.batch_out.preds = dict(outputs)

        # ----- batch end
        # aggregate to epoch storage (CPU detach)
        self._aggregate_batch_predictions()

    # ----- private method
    # ----- batch extraction
    def _parse_batch(self) -> None:
        '''
        Extract input, targets, and domain tensors from current batch.

        Expects a `(x, y, domain)` tuple in `self.state.batch_cxt.batch`
            where:
            - x: float tensor of shape [B, S, H, W].
            - y: int/long tensor of shape [B, S, H, W] or empty [B, 0]
                for inference.
            - domain: dict[str, torch.Tensor] or empty dict.

        Populates `self.state.batch_cxt.x`, `.y_dict`, and `.domain`.
        '''

        # get device
        device = self.device
        # make sure the batch is properly populated and parse
        assert self.state.batch_cxt.batch is not None
        x, y, domain = self.state.batch_cxt.batch

        # check each data and move to device
        # x should always be present
        assert isinstance(x, torch.Tensor) and x.ndim == 4 # shape [B, S, H, W]
        x = x.to(device)
        # whether label is a placeholder
        has_label = y.numel() > 0
        if has_label:
            assert isinstance(y, torch.Tensor) and y.ndim == 4 # shape [B, S, H, W]
            # x and y should have the same batch size and h*w, slice might differ
            assert x.shape[0] == y.shape[0] and x.shape[-2:] == y.shape[-2:]
            y = y.to(device)
        # domain can be an empty dict or a dict[str, torch.Tensore]
        work_domain = {}
        if domain:
            # each domain can be a tensor with the same batch size or None
            for k, v in domain.items():
                if isinstance(v, torch.Tensor):
                    assert v.shape[0] == x.shape[0]
                    work_domain[k] = v
                else:
                    work_domain[k] = None
            work_domain = {k: v.to(device) for k, v in work_domain.items()}
        else:
            work_domain = {}

        # heads: fall back to all available heads if activce heads not provided
        if self.state.heads.active_heads is None:
            self.state.heads.active_heads = self.state.heads.all_heads
        # aliases
        active_heads = self.state.heads.active_heads
        all_heads = self.state.heads.all_heads

        # precompute head index mapping
        hmap = {name: i for i, name in enumerate(all_heads)}
        # validate active heads
        m = set(active_heads) - set(all_heads)
        if m:
            raise KeyError(f'Active heads not found in all_heads: {m}')

        # extract y slices for currently active heads (empty for inference)
        # NOTE: assumes y is [B, S, H, W] and head index maps to axis=1
        y_dict = {h: y[:, hmap[h], ...] for h in active_heads} if has_label else {}

        # assign to context
        self.state.batch_cxt.x = x
        self.state.batch_cxt.y_dict = y_dict
        self.state.batch_cxt.domain = work_domain

    # ----- context setup
    def _autocast_ctx(self):
        '''Autocast context for training based on precision setting.'''
        device_type = self.device
        # pick dtype; feel free to prefer bf16 if supported:
        dtype = torch.bfloat16 if device_type == 'cpu' else torch.float16
        if self.use_amp:
            return torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=self.use_amp
            )
        return contextlib.nullcontext()

    def _val_ctx(self):
        '''Inference mode and autocast context for validation phase.'''
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(self._autocast_ctx())
        return stack

    # ----- training phase
    def _compute_loss(self) -> None:
        '''
        Compute total and per-head losses for the current batch.

        This method computes loss values using the active head loss
        modules and writes the following fields into runtime state:

        - state.batch_out.total_loss
        - state.batch_out.head_loss

        The interpretation, aggregation, backward pass, and optimization
        associated with these loss values are handled externally by the
        Trainer.
        '''

        # sanity
        assert self.state.batch_cxt.y_dict is not None
        assert self.state.heads.active_hspecs is not None
        assert self.state.heads.active_hloss is not None
        # call loss function
        total, perhead = batch.multihead_loss(
            multihead_preds=self.state.batch_out.preds,
            multihead_targets=self.state.batch_cxt.y_dict,
            features=self.state.batch_cxt.x, # image array as the features
            headspecs=self.state.heads.active_hspecs,
            headlosses=self.state.heads.active_hloss
        )
        # pass to state
        self.state.batch_out.total_loss = total
        self.state.batch_out.head_loss = perhead

    # ----- validation phase
    def _update_conf_matrix(self) -> None:
        '''
        Update per-head metric accumulators.

        This method performs incremental metric updates only (e.g.
        confusion matrix accumulation). Final metric computation and
        reporting are owned by the Trainer or Evaluator at phase
        boundaries.
        '''

        # sanity
        # head metrics is set
        assert self.state.heads.active_hmetrics is not None
        # data contains y labels
        assert self.state.batch_cxt.y_dict is not None
        # get predictions and targets
        preds = self.state.batch_out.preds
        targets = self.state.batch_cxt.y_dict
        for head, logits in preds.items():
            parent = self.parent_map.get(head)
            parent_1b = targets.get(parent) if parent is not None else None
            # retrieve head metric calculator
            metrics_module = self.state.heads.active_hmetrics[head]
            metrics_module.update(
                logits,                     # 0-based
                targets[head],              # 1-based
                parent_raw_1b=parent_1b     # 1-based (keyword arg)
            )

    # ----- inference phase
    def _aggregate_batch_predictions(self):
        '''
        Aggregate inference batch predictions into runtime storage.

        This method maps per-batch predictions into epoch-level inference
        buffers based on spatial layout information. Interpretation and
        visualization of these results are handled externally.
        '''

        # inference context alias
        ctx = self.state.summary.infer_context

        # get through each head and attach preds to corresponding block-heads
        for head, logits in self.state.batch_out.preds.items():

            # logits: [B, C, Hp, Wp] -> preds: # [B, Hp, Wp]
            preds = torch.argmax(logits, dim=1)
            # get preds placed according to locations from this batch
            mapped: dict[tuple[int, int], torch.Tensor] = {}
            # place patches to locations
            for i, pred in enumerate(preds):
                # global patch index
                patch_idx = self.state.batch_cxt.pidx_start + i
                # block position
                block_idx = patch_idx // ctx.patch_per_blk
                block_row, block_col = divmod(block_idx, ctx.block_columns)
                # patch position inside block
                p_in_blk = patch_idx % ctx.patch_per_blk
                patch_row, patch_col = divmod(p_in_blk, ctx.patch_per_dim)
                # map pred to patch-grid coordinates
                mapped[(
                    block_col * ctx.patch_per_dim + patch_col,
                    block_row * ctx.patch_per_dim + patch_row
                )] = pred

            # pull existing epoch-level maps (persist across batches)
            # ensure per-head dict exists, then mutate
            ctx.maps.setdefault(head, {}).update(mapped)
            # Update (merge) mapping for this head
            # Keys are (col, row) in patch-grid coords
