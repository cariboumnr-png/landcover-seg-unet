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
Execution core for session engines.

This module defines the batch-level execution engine used by higher-level
training, validation, and inference controllers.

The execution core is responsible for:
- Parsing input batches into runtime tensors
- Running model forward passes under the correct context
- Computing losses and metrics
- Updating runtime state deterministically
- Emitting lifecycle callbacks for observation and side effects

The execution core deliberately does NOT:
- Own optimizer or scheduler logic
- Control epoch structure or phase transitions
- Decide logging intervals or checkpointing
- Perform experiment-level orchestration

It answers the question:
    "What happened in this **batch**?"

Higher-level controllers (e.g. Trainer, Runner) answer:
    "What should we do next?"
'''

# standard imports
import contextlib
# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.engine.core as engine_core

class BatchExecutionEngine:
    '''
    Batch execution engine for session workflows.

    This class encapsulates all batch-level execution mechanics shared
    across training, validation, and inference, including:

    - Parsing input batches into tensors
    - Managing precision and evaluation contexts
    - Running model forward passes
    - Computing losses and metrics
    - Aggregating inference outputs
    - Mutating runtime state
    - Emitting lifecycle callback hooks

    EngineCore executes exactly one batch at a time and is intentionally
    ignorant of epoch structure, optimization policy, logging cadence,
    and experiment control flow.

    It is designed to be driven by a higher-level controller (e.g.
    Trainer) that:
    - Iterates over epochs and dataloaders
    - Owns optimizer and scheduler behavior
    - Interprets and consumes runtime state
    - Decides when to log, checkpoint, or stop

    In other words, EngineCore defines *execution correctness*, not
    *training policy*.
    '''

    def __init__(
        self,
        model: core.MultiheadModelLike,
        components: common.TrainerComponentsLike,
        config: common.TrainerConfigShape,
        device: str,
        **kwargs
    ):
        '''
        Initialize the engine core with components and configuration.

        A `RuntimeState` object is created and callback classes are to
        wired to the core.

        Args:
            components: Dataclass of core components.
            config: Dataclass of runtime configuration.
            device: Device string (e.g., 'cpu'/'cuda'/or 'cuda:0').
            kwargs:
                runtime convenience flags
                - skip_log: bool
        '''

        # get model and trainer components
        self.model = model
        self.comps = components
        # move model to device
        self.device = device
        self.model.to(self.device)
        # get model runtime config
        self.config = config
        # init the runtime state
        self.state = self._init_state()
        # setup callback classes
        self._setup_callbacks(kwargs.get('skip_log', True))

    # ----- Public Method
    def run_train_batch(self, bidx: int, batch: tuple) -> None:
        '''
        Execute one training batch.

        This method performs batch-level execution, including:
        - parsing inputs into runtime tensors
        - running the forward pass under the correct context
        - computing losses and updating runtime state
        - emitting lifecycle callback hooks for observation

        Callback hooks are emitted as **semantic execution markers**:
        - `on_train_batch_begin`
        - `on_train_batch_forward`
        - `on_train_batch_compute_loss`
        - `on_train_batch_end`
        but do not invoke or control execution logic.

        See `./callback/phase_train.py` for training-phase callback
        observers.
        '''

        print(f'Training Batch_{bidx}', end='\r', flush=True)

        # ----- batch start
        # get new batch and parse into x, y_dict and domain
        self._parse_batch()
        self._emit('on_train_batch_begin', bidx, batch)

        # ----- batch forward
        # get x and domain (optional)
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with autocast context
        with self._autocast_ctx():
            predictions = self.model.forward(x, **domain)
        # assign predictions to batch output
        self.state.batch_out.preds = dict(predictions)
        self._emit('on_train_batch_forward')

        # ----- batch compute loss
        # compute loss with autocast context
        with self._autocast_ctx():
            self._compute_loss()
        self._emit('on_train_batch_compute_loss')

        # ----- batch end
        self._emit('on_train_batch_end')

    def run_validate_batch(self, bidx: int, batch: tuple) -> None:
        '''
        Execute one validation batch.

        This method performs batch-level execution, including:
        - parsing inputs into runtime tensors
        - running the forward pass under evaluation / inference context
        - updating validation metrics and runtime state
        - emitting lifecycle callback hooks for observation

        Callback hooks are emitted as **semantic execution markers**:
        - `on_validation_batch_begin`
        - `on_validation_batch_forward`
        - `on_validation_batch_end`
        but do not invoke or control execution logic.

        See `./callback/phase_val.py` for validation-phase callback
        observers.
        '''

        # ----- batch start
        # get new batch and parse into x, y_dict and domain
        self._parse_batch()
        self._emit('on_validation_batch_begin', bidx, batch)

        # ----- batch forward
        # get x and domain (optional)
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with autocast context
        with self._val_ctx():
            outputs = self.model.forward(x, **domain)
        self.state.batch_out.preds = dict(outputs)
        self._emit('on_validation_batch_forward')

        # ----- batch end
        # update per-head confusion matrix
        self._update_conf_matrix()
        self._emit('on_validation_batch_end')

    def run_infer_batch(self, bidx: int, batch: tuple):
        '''
        Execute one inference batch.

        This method performs batch-level execution, including:
        - parsing inputs into runtime tensors (inputs + optional domain)
        - running the forward pass under inference context
        - aggregating prediction results into inference runtime state
        - emitting lifecycle callback hooks for observation

        Callback hooks are emitted as **semantic execution markers**:
        - `on_inference_batch_begin`
        - `on_inference_batch_forward`
        - `on_inference_batch_end`
        but do not invoke or control execution logic.

        See `./callback/phase_infer.py` for inference-phase callback
        observers.
        '''

        # ----- batch start
        # parse the batch (x, domain), y may be empty [B, 0]
        self._parse_batch()
        self._emit('on_inference_batch_begin', bidx, batch)

        # ----- batch forward
        # get x and (optional) domain
        x = self.state.batch_cxt.x
        domain = self.state.batch_cxt.domain
        # forward with inference + autocast (same style as validation)
        with self._val_ctx():
            outputs = self.model.forward(x, **domain)
        # store raw predictions to batch output
        self.state.batch_out.preds = dict(outputs)
        self._emit('on_inference_batch_forward')

        # ----- batch end
        # aggregate to epoch storage (CPU detach)
        self._aggregate_batch_predictions()
        self._emit('on_inference_batch_end')

    # ----- private method
    def _init_state(self) -> engine_core.RuntimeState:
        '''Instantiate the runtime state aligned with trainer config.'''

        # instantiate a state
        state = engine_core.init_state()
        # state - full batch size:
        state.batch_cxt.batch_size_full = self.comps.dataloaders.meta.batch_size
        # state - heads
        state.heads.all_heads = list(self.comps.headspecs.as_dict().keys())
        # state = optimization
        state.optim.scaler = torch.GradScaler(
            device=self.device,
            enabled=self.config.precision.use_amp
        )
        # if test dataset if provided, setup inference context
        if self.comps.dataloaders.test:
            # resolve patch-block layout
            per_blk = self.comps.dataloaders.meta.patch_per_blk
            per_dim = int(per_blk ** 0.5)
            assert per_dim * per_dim == per_blk, 'patch_per_blk must be square'
            state.epoch_sum.infer_ctx.patch_per_blk = per_blk
            state.epoch_sum.infer_ctx.patch_per_dim = per_dim
            # resolve block col/row numbers
            blk_col, blk_row = self.comps.dataloaders.meta.test_blks_grid
            state.epoch_sum.infer_ctx.block_columns = blk_col
            # resolve patch col/row numbers
            pch_col, pch_row = (blk_col * per_dim, blk_row * per_dim)
            state.epoch_sum.infer_ctx.patch_grid_shape = pch_col, pch_row
        # return
        return state

    def _setup_callbacks(self, skip_logs: bool) -> None:
        '''Pass current trainer instance to all callback classes.'''

        for callback in self.comps.callbacks:
            callback.setup(self, skip_logs)

    def _emit(self, hook: str, *args, **kwargs) -> None:
        '''
        Invoke a named hook from callbacks with the provided arguments.

        Args:
            hook: Hook method to call (e.g., 'on_train_batch_begin').
            *args: Positional arguments passed to the callback method.
            **kwargs: Keyword arguments passed to the callback method.
        '''

        for callback in self.comps.callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args, **kwargs)

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
        if self.config.precision.use_amp:
            return torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=self.config.precision.use_amp
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

        Writes `total_loss` and `head_loss` into `self.state.batch_out`.
        '''

        # sanity
        assert self.state.batch_cxt.y_dict is not None
        assert self.state.heads.active_hspecs is not None
        assert self.state.heads.active_hloss is not None
        # call loss function
        total, perhead = engine_core.multihead_loss(
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
        '''Update confusion matrix in-place at validation batch end.'''

        # sanity
        # head metrics is set
        assert self.state.heads.active_hmetrics is not None
        # data contains y labels
        assert self.state.batch_cxt.y_dict is not None
        # get predictions and targets
        preds = self.state.batch_out.preds
        targets = self.state.batch_cxt.y_dict
        for head, logits in preds.items():
            parent = self.comps.headspecs[head].parent_head
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
        '''Accumulate batch predictions a class map.'''

        # inference context alias
        ctx = self.state.epoch_sum.infer_ctx

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
