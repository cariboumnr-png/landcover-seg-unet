'''
Multi-head training orchestration for hierarchical segmentation.

This module defines `MultiHeadTrainer`, a callback-driven trainer that
wires together model, data loaders, optimization, and per-head
specifications for multi-head hierarchical segmentation/classification
tasks.

Key features:
- Epoch-level training and validation with callback hooks.
- Dynamic control of active/frozen heads and per-head class exclusion.
- Mixed-precision contexts and gradient scaling.
- Aggregated logging for total and per-head losses, and per-head IoU
    metrics.

See also:
- training.trainer (components, configs, and runtime state)
- callback modules under ./callback for phase-specific hooks
'''

# standard imports
import contextlib
import copy
# third-party imports
import torch
# local imports
import training.trainer
import utils

class MultiHeadTrainer:
    '''
    Trainer for multi-head hierarchical segmentation.

    Responsibilities:
    - Wire up model, data loaders, optimizer/scheduler, head specs,
        loss, and metrics.
    - Run epoch-level training and validation via callback hooks.
    - Set/reset active/frozen training heads; mask specific classes for
        loss/metrics.
    - Aggregate logs and track best metric.

    This trainer delegates workflow steps to callback classes (see
    `callback` modules) but retains core logic such as batch parsing,
    context management, and phase-specific helpers.
    '''

    def __init__(
            self,
            components: training.trainer.TrainerComponents,
            config: training.trainer.RuntimeConfig,
            device: str
        ):
        '''
        Initialize trainer with components and configuration.

        A `RuntimeState` object is created and callback classes are to
        wired to the trainer.

        Args:
            components: Dataclass of trainer components (see
                `./comps.py`).
            config: Dataclass of runtime configuration. (see
                `./config.py`).
            device (str): Device string (e.g., 'cpu'/'cuda'/or 'cuda:0')
                applied at trainer level.
        '''

        # parse arguments
        self.comps = components
        self.config = config
        self.device = device
        # move model to device
        self.model.to(self.device)
        # init the runtime state
        self.state = self._init_state()
        # setup callback classes
        self._setup_callbacks()

# -------------------------------Public  Methods-------------------------------
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        '''
        Train for one epoch and return an epoch-level training log.

        Orchestrated by callback hooks that invoke private helpers:
        - `on_train_epoch_begin`
        - `on_train_batch_begin`
        - `on_train_batch_forward`
        - `on_train_backward`
        - `on_train_before_optimizer_step`
        - `on_train_optimizer_step`
        - `on_train_batch_end`
        - `on_train_epoch_end`

        See `./training/callback/phase_train.py` for phase-specific hook
        implementations.

        Returns:
            A dictionary of averaged training metrics (e.g.,
            total/per-head loss).
        '''

        # train phase begin:
        # - set model to .train()
        # - reset train loss/logs
        self._emit('on_train_epoch_begin', epoch)

        # interate through training data batches
        assert self.dataloaders.train, 'Training dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.train, start=1):
            print(f'Training Batch_{bidx}', end='\r', flush=True)

            # batch start
            # - get new batch and parse into x, y_dict and domain
            # - reset optimizer gradients
            self._emit('on_train_batch_begin', bidx, batch)

            # batch forward
            # - forward the model with x and domain
            # - autocast context handled
            self._emit('on_train_batch_forward')

            # batch compute loss
            # - compute total and per-head loss
            # - autocast context handled
            self._emit('on_train_batch_compute_loss')

            # batch backward
            self._emit('on_train_backward')

            # gradient clipping
            self._emit('on_train_before_optimizer_step')

            # optimizer step
            self._emit('on_train_optimizer_step')

            # batch end
            # - update train loss and log dict
            self._emit('on_train_batch_end')

        # train phase end
        # - update logs and loss (total/per-head) for the epoch
        self._emit('on_train_epoch_end')
        return self.state.epoch_sum.train_logs.head_losses

    def validate(self) -> dict[str, dict]:
        '''
        Validate on the validation set and return epoch-level metrics.

        Orchestrated by callback hooks that invoke private helpers:
        - `on_validation_begin`
        - `on_validation_batch_begin`
        - `on_validation_batch_forward`
        - `on_validation_batch_end`
        - `on_validation_end`

        See `.training/callback/phase_val.py` for phase-specific hook
        implementations.

        Returns:
            A mapping from head name to its validation metrics (dict).
        '''

        # val phase start
        # - set model to .eval()
        # - reset head confusion matrices
        # - reset validation loss/logs
        self._emit('on_validation_begin')

        # iterate through validation dataset
        assert self.dataloaders.val, 'Validation dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.val, start=1):

            # batch start
            # - get new batch and parse into x, y_dict and domain
            self._emit('on_validation_batch_begin', bidx, batch)

            # batch forward
            # - forward the model with x and domain
            # - autocast context handled
            # - use torch.inference_mode()
            self._emit('on_validation_batch_forward')

            # batch end
            # - update per-head confusion matrix from outputs and y_dict
            self._emit('on_validation_batch_end')

        # val phase end
        # - calculate per-head IoU related metrics for and update the log dict
        self._emit('on_validation_end')
        return self.state.epoch_sum.val_logs.head_metrics

    def infer(self, out_dir: str):
        '''
        Production inference over dataloaders.infer (images only).
        Returns: head -> {block_id -> full_map_tensor}
        '''

        # infer phase start
        # - set model to .eval()
        # - clear inference output mapping (batch -> preds)
        self._emit('on_inference_begin')

        # iterate through inference dataset
        assert self.dataloaders.test, 'Inference dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.test, start=1):

            # batch start
            # - get new batch and parse into x, y_dict (empty dict) and domain
            self._emit('on_inference_batch_begin', bidx, batch)

            # batch forward
            # - forward the model with x and domain
            # - autocast context handled
            # - use torch.inference_mode()
            self._emit('on_inference_batch_forward')

            # batch end
            # - aggregate predictions to epoch storage
            self._emit('on_inference_batch_end')

        # inference phase end
        self._emit('on_inference_end', out_dir)

    def set_head_state(
            self,
            active_heads: list[str] | None=None,
            frozen_heads: list[str] | None=None,
            excluded_cls: dict[str, tuple[int, ...]] | None=None
        ) -> None:
        '''
        Set active/frozen heads and per-head class exclusions.

        Side effects:
            - Updates model active/frozen heads.
            - Deep-copies and installs per-head specs, loss, and metrics
                into `self.state`.
            - Applies per-head class exclusions to specs and metrics.

        Args:
            active_heads: Heads to activate. Defaults to all heads when
                set to `None`.
            frozen_heads: Heads to freeze (if provided).
            excluded_cls: Mapping of head -> tuple of class indices to
                exclude from loss and validation metrics.
        '''

        # if no active heads provided, make all heads active
        if active_heads is None:
            active_heads = self.state.heads.all_heads

        # set active and frozen heads
        self.state.heads.active_heads = active_heads
        self.state.heads.frozen_heads = frozen_heads

        # set active heads at model
        self.model.set_active_heads(active_heads)
        # set active heads specs
        self.state.heads.active_hspecs = {
            k: copy.deepcopy(self.headspecs[k]) for k in active_heads
        }
        # set loss module for active heads
        self.state.heads.active_hloss = {
            k: copy.deepcopy(self.headlosses[k]) for k in active_heads
        }
        # set metric module for active heads
        self.state.heads.active_hmetrics = {
            k: copy.deepcopy(self.headmetrics[k]) for k in active_heads
        }

        # set frozen heads to model if provided
        if frozen_heads is not None:
            self.model.set_frozen_heads(frozen_heads)

        # set excluded classes to active heads
        if excluded_cls is not None:
            for h in active_heads:
                excl = excluded_cls.get(h)
                if excl is not None:
                    self.state.heads.active_hspecs[h].set_exclude(excl)
                    self.state.heads.active_hmetrics[h].exclude_class_1b = excl

    def reset_head_state(self):
        '''
        Reset runtime training heads.

        Side effects:
        - Calls `model.reset_heads()`.
        - Clears active/frozen heads and related per-head modules.
        '''

        self.model.reset_heads()
        self.state.heads.active_heads = None
        self.state.heads.frozen_heads = None
        self.state.heads.active_hspecs = None
        self.state.heads.active_hloss = None
        self.state.heads.active_hmetrics = None

    def predict(
            self,
            x: torch.Tensor,
            mode: str
        ) -> dict[str, torch.Tensor]:
        '''
        Run a simple head-aware inference pass.

        For segmentation heads, returns argmax class maps; for
        classification heads, returns top-1 class indices; for other
        heads, returns raw outputs.

        Args:
            x: Input tensor, moved to trainer device.
            mode: Determins the return type.

        Returns:
            A mapping from head name to prediction tensor.
        '''

        # set model to evaluation mode and move to device
        self.model.eval()
        x = x.to(self.device)

        # get outputs in proper context
        with torch.inference_mode(), self._autocast_ctx():
            outputs = self.model.forward(x)  # raw logits per head

        # get prediction by mode and return
        preds = {}
        for head, logits in outputs.items():
            if mode == 'seg':       # segmentation head
                preds[head] = torch.argmax(logits, dim=1)  # [B, H, W]
            elif mode == 'cls':     # classification head
                probs = torch.softmax(logits, dim=1)
                preds[head] = torch.topk(probs, k=1).indices  # top-1 class
            elif mode == 'raw':      # raw values
                preds[head] = logits
            else:
                raise ValueError(
                    f'Invalid mode: {mode}: must be in ["seg", "cls", "raw"]'
                )
        return preds

    # ----------------------------internal methods----------------------------
    # runtime state initialization
    def _init_state(self) -> training.trainer.RuntimeState:
        '''Instantiate the runtime state aligned with trainer config.'''

        # instantiate a state
        state = training.trainer.RuntimeState()
        # state - full batch size:
        state.batch_cxt.batch_size_full = self.dataloaders.meta.batch_size
        # state - heads
        state.heads.all_heads = list(self.headspecs.as_dict().keys())
        # state = optimization
        state.optim.scaler = torch.GradScaler(
            device=self.device,
            enabled=self.config.precision.use_amp
        )
        # return
        return state

    # callback classes setup
    def _setup_callbacks(self) -> None:
        '''Pass current trainer instance to all callback classes.'''

        for callback in self.callbacks:
            callback.setup(self)

    # callback callers
    def _emit(self, hook: str, *args, **kwargs) -> None:
        '''
        Invoke a named hook from callbacks with the provided arguments.

        Args:
            hook: Hook method to call (e.g., 'on_train_batch_begin').
            *args: Positional arguments passed to the callback method.
            **kwargs: Keyword arguments passed to the callback method.
        '''

        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args, **kwargs)

    # -----------------------helpers called by callbacks----------------------
    # batch extraction
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

    # context setup
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

    # training phase
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
        total, perhead = training.trainer.multihead_loss(
            multihead_preds=self.state.batch_out.preds,
            multihead_targets=self.state.batch_cxt.y_dict,
            headspecs=self.state.heads.active_hspecs,
            headlosses=self.state.heads.active_hloss
        )
        # pass to state
        self.state.batch_out.total_loss = total
        self.state.batch_out.head_loss = perhead

    def _clip_grad(self):
        '''Clip gradients by global norm when set.'''
        if self.config.optim.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optim.grad_clip_norm
            )

    def _update_train_logs(self, flush: bool=False):
        '''
        Average losses and update per-head loss logs at intervals.

        Set `flush=True` to flush results at end of a training epoch.
        '''

        # get current batch id
        bidx = self.state.batch_cxt.bidx
        # create log dict
        logs = {}
        # update log at interval
        if flush or bidx % self.config.schedule.logging_interval == 0:
            # average total loss so far
            avg_loss = self.state.epoch_sum.train_loss / max(1, bidx)
            logs['Total_Loss'] = avg_loss
            # per-head losses
            for h, v in self.state.batch_out.head_loss.items():
                logs[f'Head_Loss_{h}'] = float(v)
            # extras - lr
            logs['LR'] = self.optimization.optimizer.param_groups[0]['lr']
            # assgin to state dict
            self.state.epoch_sum.train_logs.head_losses = logs
        else:
            self.state.epoch_sum.train_logs.updated = False # reset flag

        # if logs are updated: provide a printer friendly text and return flag
        if logs:
            self.state.epoch_sum.train_logs.updated = True
            text_list = [f'{k}: {v:.4f}' for k, v in logs.items()]
            text = f'b{bidx:04d} | ' + '|'.join(text_list)
            self.state.epoch_sum.train_logs.head_losses_str = text

    # validation phase
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
            parent = self.headspecs[head].parent_head
            parent_1b = targets.get(parent) if parent is not None else None
            # retrieve head metric calculator
            metrics_module = self.state.heads.active_hmetrics[head]
            metrics_module.update(
                p0=logits,                              # 0-based
                t1=targets[head],                       # 1-based
                parent_raw_1b=parent_1b                 # 1-based (keyword arg)
            )

    def _compute_iou(self) -> None:
        '''
        Compute IoU at the end of validation phase.

        Writes metrics to `self.state.epoch_sum`.
        '''

        # sanity
        assert self.state.heads.active_hmetrics is not None
        val_logs: dict[str, dict] = {}
        val_logs_text: dict[str, list[str]] = {}
        # calculate IoU related metrics for each head
        for head, metrics_module in self.state.heads.active_hmetrics.items():
            metrics_module.compute() # final metrics from batch accumulations
            val_logs[head] = metrics_module.metrics_dict
            val_logs_text[head] = metrics_module.metrics_text
        self.state.epoch_sum.val_logs.head_metrics = val_logs
        self.state.epoch_sum.val_logs.head_metrics_str = val_logs_text

    def _track_metrics(self) -> None:
        '''Track the best metric and count patience epochs.'''

        # get metric from validation metrics dictionary
        track_head = self.config.monitor.head
        val = self.state.epoch_sum.val_logs.head_metrics[track_head]
        met = val['ac_mean'] if val['has_active'] in val else val['mean']

        # at the end of the first epoch
        if self.state.progress.epoch == 1:
            self.state.metrics.last_value = 0.0
            self.state.metrics.curr_value = met
        else:
            self.state.metrics.last_value = self.state.metrics.curr_value
            self.state.metrics.curr_value = met

        # determine the best metrics
        delta = self.config.schedule.min_delta or 0.0 # None -> 0.0 (no delta)
        assert delta >= 0.0 # sanity
        # maximize tracking metrics
        if self.config.monitor.mode == 'max':
            # update tracking numbers
            if met >= self.state.metrics.best_value + delta:
                self.state.metrics.best_value = met
                self.state.metrics.best_epoch = self.state.progress.epoch
                self.state.metrics.patience_n = 0 # reset patience
            # otherwise increment patience counter
            else:
                self.state.metrics.patience_n += 1

    # inference phase
    def _setup_infer_context(self):
        '''Setup context for the inference stage.'''

        per_blk = self.dataloaders.meta.patch_per_blk
        blk_grid = self.dataloaders.meta.test_blks_grid # shape (col, row)
        self.state.epoch_sum.infer_ctx.setup(per_blk, blk_grid)

    def _aggregate_batch_predictions(self):
        '''Convert predictions of the current batch into a class map'''

        # pull existing epoch-level maps (persist across batches)
        epoch_maps: dict[str, dict[tuple[int, int], torch.Tensor]] | None
        epoch_maps = getattr(self.state.epoch_sum.infer_ctx, 'maps', None)
        if epoch_maps is None:
            epoch_maps = {}
            self.state.epoch_sum.infer_ctx.maps = epoch_maps

        # get through each head and attach preds to corresponding block-heads
        for head, logits in self.state.batch_out.preds.items():
            # logits: [B, C, Hp, Wp] -> preds: # [B, Hp, Wp]
            preds = torch.argmax(logits, dim=1)
            # get preds placed according to locations from this batch
            placed = self.__place_patches(preds)
            # ensure per-head dict exists, then accumulate
            if head not in epoch_maps:
                epoch_maps[head] = {}
            # Update (merge) placements for this head
            # Keys are (col, row) in patch-grid coords
            epoch_maps[head].update(placed)

    def __place_patches(
        self,
        preds: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        '''
        Map batch patch predictions to global (x, y) pixel positions.

        Args:
            preds: [B, Hp, Wp]
            patch_per_blk: patches per block
            patch_per_dim: patches per dimension (p = n*n, square)
            start_patch_idx: global patch index for preds[0]

        Returns:
            {(x, y): patch_tensor}
        '''

        # get values from contxt
        patch_per_blk = self.state.epoch_sum.infer_ctx.patch_per_blk
        patch_per_dim = self.state.epoch_sum.infer_ctx.patch_per_dim
        cols_total = self.state.epoch_sum.infer_ctx.block_columns

        # output dict
        out: dict[tuple[int, int], torch.Tensor] = {}
        # place patches to locations
        for i in range(preds.shape[0]):
            # global patch tracker
            global_patch_i = self.state.batch_cxt.pidx_start + i
            # which block?
            global_block_i = global_patch_i // patch_per_blk
            block_col = global_block_i % cols_total
            block_row = global_block_i // cols_total
            # patch position inside bloc
            p_in_blk = global_patch_i % patch_per_blk
            patch_col = p_in_blk % patch_per_dim
            patch_row = p_in_blk // patch_per_dim
            # patch-grid coordinates
            col = block_col * patch_per_dim + patch_col
            row = block_row * patch_per_dim + patch_row
            out[(col, row)] = preds[i]
        # return
        return out

    def _preview_monitor_head(self, out_dir: str) -> None:
        '''doc'''

        utils.export_previews(
            self.state.epoch_sum.infer_ctx.maps,
            self.state.epoch_sum.infer_ctx.patch_grid_shape,
            out_dir=out_dir,
            heads=[self.config.monitor.head] # as list
        )

    # -------------------------convenience properties-------------------------
    @property
    def model(self):
        '''Shortcut to model.'''
        return self.comps.model

    @property
    def dataloaders(self):
        '''Shortcut to dataloaders.'''
        return self.comps.dataloaders

    @property
    def headspecs(self):
        '''Shortcut to headspecs.'''
        return self.comps.headspecs

    @property
    def headlosses(self):
        '''Shortcut to headlosses.'''
        return self.comps.headlosses

    @property
    def headmetrics(self):
        '''Shortcut to headmetrics.'''
        return self.comps.headmetrics

    @property
    def optimization(self):
        '''Shortcut to optimization.'''
        return self.comps.optimization

    @property
    def callbacks(self):
        '''Shortcut to callbacks.'''
        return self.comps.callbacks
