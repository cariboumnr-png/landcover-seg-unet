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
Multi-head training and evaluation policy for hierarchical segmentation.

This module defines `MultiHeadTrainer`, which provides epoch-level
training, validation, and inference policies on top of a shared
batch execution engine.

The trainer is responsible for:
- Defining training, validation, and inference workflows
- Managing optimizer, scheduler, and gradient handling
- Controlling active and frozen heads across phases
- Applying per-head class exclusions
- Aggregating epoch-level statistics and metrics
- Tracking best validation metrics and early-stopping state
- Emitting lifecycle callbacks as semantic markers

The trainer deliberately does NOT:
- Perform batch-level execution (forward, loss, metrics)
- Parse input batches
- Compute losses or metrics incrementally

Those responsibilities are delegated to `BatchExecutionEngine`,
which mutates shared runtime state during batch execution.

In short:
- The batch executor answers: "What happened in this batch?"
- The trainer answers: "How should batch results be used over time?"
'''

# standard imports
import copy
import typing
# third-party imports
import torch
# local imports
import landseg.session.common as common
import landseg.session.engine.core as engine_core

class MultiHeadTrainer:
    '''
    Training and evaluation policy controller.

    This class defines epoch-level behavior for training, validation,
    and inference by orchestrating:

    - Batch execution via a shared BatchExecutionEngine
    - Optimizer and scheduler behavior
    - Gradient scaling and clipping
    - Head activation, freezing, and exclusion rules
    - Epoch-level loss aggregation and metric finalization
    - Experimental metric tracking and early-stopping state

    The trainer operates on a shared RuntimeState object that persists
    across Runner, Trainer/Evaluator, and the batch execution engine.

    Batch-level mathematical execution is delegated entirely to
    BatchExecutionEngine. This class focuses exclusively on policy,
    orchestration, and interpretation of execution results.

    Lifecycle callback hooks emitted by this class serve as semantic
    markers for observation and side effects (logging, monitoring,
    visualization), but do not invoke or control execution logic.
    '''

    class Flags(typing.TypedDict):
        '''Typed training running flags (flexible).'''
        skip_log: bool
        enable_train_la: bool
        enable_val_la: bool
        enable_test_la: bool

    def __init__(
        self,
        engine: engine_core.BatchExecutionEngine,
        state: engine_core.RuntimeState,
        components: common.TrainerComponentsLike,
        config: common.TrainerConfigShape,
        device: str,
        **kwargs
    ):
        '''
        Initialize the trainer.

        The trainer is constructed with an already-initialized batch
        execution engine and a shared RuntimeState. Runtime state is not
        owned by the trainer but is interpreted and mutated according
        to training and evaluation policy.

        Args:
            engine:
                BatchExecutionEngine responsible for batch-level execution.
            state:
                Shared RuntimeState instance updated by the batch executor
                and consumed by the trainer.
            components:
                Trainer components including dataloaders, callbacks,
                optimizer, loss modules, and metric modules.
            config:
                Runtime configuration controlling training schedule,
                precision, and monitoring behavior.
            device:
                Device identifier (e.g., 'cpu', 'cuda', 'cuda:0') applied
                at the trainer level.
            kwargs:
                Runtime control flags:
                - skip_log: Disable logging callbacks
                - enable_train_la: Enable logit adjustment during training
                - enable_val_la: Enable logit adjustment during validation
                - enable_test_la: Enable logit adjustment during inference
        '''

        # get attributes from engine
        self.engine = engine
        self.model = engine.model
        self.state = state
        self.comps = components
        # move model to device
        self.device = device
        self.model.to(self.device)
        # get model runtime config
        self.config = config
        # populate runtime flags from kwargs
        self.flags = {
            'skip_log': kwargs.get('skip_log', False),
            'enable_train_la': kwargs.get('enable_train_la', False),
            'enable_val_la': kwargs.get('enable_val_la', False),
            'enable_test_la': kwargs.get('enable_test_la', False),
        }
        # setup callback classes
        for callback in self.callbacks:
            callback.setup(self, self.flags['skip_log'])

    # ----- property
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


# -------------------------------Public  Methods-------------------------------
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        '''
        Execute one training epoch and return epoch-level training logs.

        This method defines the training policy for a single epoch,
        including:

        - Model mode and logit-adjustment configuration
        - Optimizer and gradient-scaling behavior
        - Gradient clipping and optimizer stepping
        - Delegation of batch execution to BatchExecutionEngine
        - Aggregation and formatting of epoch-level loss statistics

        Batch-level execution (forward pass, loss computation, metric
        accumulation) is performed by the batch execution engine and
        reflected in shared RuntimeState.

        Lifecycle callback hooks (e.g., `on_train_epoch_begin`,
        `on_train_backward`, `on_train_optimizer_step`,
        `on_train_batch_end`) are emitted as semantic markers to allow
        observation and side effects, but do not invoke or control
        execution logic.

        Returns:
            A dictionary of averaged epoch-level training metrics
            (e.g., total and per-head losses).
        '''

        # ----- train phase begin
        # set model to train mode
        self.model.train()
        # set logit adjustment status
        self.model.set_logit_adjust_enabled(self.flags['enable_train_la'])
        self._emit('on_train_epoch_begin', epoch)

        # interate through training data batches
        assert self.dataloaders.train, 'Training dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.train, start=1):

            # batch start
            self._emit('on_train_batch_begin', bidx, batch)
            # reset optimizer gradient
            self.comps.optimization.optimizer.zero_grad(set_to_none=True)

            # delegate batch to engine (forward and compute loss)
            self.engine.run_train_batch(bidx)

            # batch backward
            loss = self.state.batch_out.total_loss
            if self.config.precision.use_amp:
                self.state.optim.scaler.scale(loss).backward()
            else:
                loss.backward()
            self._emit('on_train_backward')

            # gradient clipping
            optimizer = self.comps.optimization.optimizer
            # unscale if use AMP
            if self.config.precision.use_amp:
                self.state.optim.scaler.unscale_(optimizer)
            self._clip_grad()
            self._emit('on_train_before_optimizer_step')

            # optimizer step
            optimizer = self.comps.optimization.optimizer
            # use AMP
            if self.config.precision.use_amp:
                self.state.optim.scaler.step(optimizer)
                self.state.optim.scaler.update() # update scaler
            # no AMP
            else:
                self.comps.optimization.optimizer.step()
            self._emit('on_train_optimizer_step')

            # batch end
            # accumulate batch loss to epoch-level total loss
            batch_loss = float(self.state.batch_out.total_loss.detach().item())
            self.state.epoch_sum.train_loss += batch_loss
            # update train logs if at interval (decided by trainer method)
            self._update_train_logs(flush=False)
            self._emit('on_train_batch_end')

        # train phase end
        # - update logs and loss (total/per-head) for the epoch
        self._update_train_logs(flush=True)
        self._emit('on_train_epoch_end')
        return self.state.epoch_sum.train_logs.head_losses

    def validate(self) -> dict[str, dict]:
        '''
        Execute validation over the validation dataset and return
        epoch-level metrics.

        This method defines the validation policy, including:

        - Model evaluation mode and logit-adjustment configuration
        - Delegation of batch execution to the batch execution engine
        - Finalization and formatting of validation metrics
        - Tracking of best metrics and patience state

        Batch-level metric accumulation (e.g., confusion matrices) is
        performed by the batch execution engine during validation batches.
        Epoch-level metric computation and interpretation are performed
        here.

        Lifecycle callback hooks (e.g., `on_validation_begin`,
        `on_validation_batch_begin`, `on_validation_end`) are emitted
        as semantic markers for observation and side effects.

        Returns:
            A mapping from head name to its finalized validation metrics.
        '''

        # val phase start
        # set model to evaluation mode
        self.model.eval()
        # set logit adjustment status
        self.model.set_logit_adjust_enabled(self.flags['enable_val_la'])
        self._emit('on_validation_begin')

        # iterate through validation dataset
        assert self.dataloaders.val, 'Validation dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.val, start=1):

            # batch start
            self._emit('on_validation_batch_begin', bidx, batch)

            # delegate to batch executor
            self.engine.run_validate_batch()

        # val phase end
        # compute iou
        self._compute_iou()
        # update experimental level metrics
        self._track_metrics()
        self._emit('on_validation_end')
        return self.state.epoch_sum.val_logs.head_metrics

    def infer(self, out_dir: str):
        '''
        Execute inference over the test dataset.

        This method defines inference policy, including:

        - Model evaluation mode and logit-adjustment configuration
        - Delegation of batch execution to the batch execution engine
        - Emission of inference lifecycle events for side effects

        Batch-level inference execution and aggregation are handled by
        the batch execution engine. This method does not interpret or
        post-process inference results directly.
        '''

        # infer phase start
        # set model to evaluation mode
        self.model.eval()
        # set logit adjustment status
        self.model.set_logit_adjust_enabled(self.flags['enable_test_la'])
        self._emit('on_inference_begin')

        # iterate through inference dataset
        assert self.dataloaders.test, 'Inference dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.test, start=1):

            # batch start
            self._emit('on_inference_batch_begin', bidx, batch)

            # delegate to batch executor
            self.engine.run_infer_batch()

        # inference phase end
        # - produce a preview image if the test blocks grid is valid
        self._emit('on_inference_end', out_dir)

    def set_head_state(
        self,
        active_heads: list[str] | None=None,
        frozen_heads: list[str] | None=None,
        excluded_cls: dict[str, list[int]] | None=None
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
                    self.state.heads.active_hspecs[h].set_exclude(tuple(excl))
                    self.state.heads.active_hmetrics[h].exclude_class_1b = tuple(excl)

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

    def config_logit_adjustment(
        self,
        *,
        enable_train_logit_adjustment: bool,
        enable_val_logit_adjustment: bool,
        enable_test_logit_adjustment: bool,
        **kwargs
    ) -> None:
        '''
        Simple helper to set logit adjustment use flags.
        '''

        # assign flags
        self.flags['enable_train_la'] = enable_train_logit_adjustment
        self.flags['enable_val_la'] = enable_val_logit_adjustment
        self.flags['enable_test_la'] = enable_test_logit_adjustment
        # implemented for signature flexibility
        if kwargs:
            pass

    # ----- callback signal emitter
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

    # ----- training phase
    def _clip_grad(self):
        '''Clip gradients by global norm when set.'''
        if self.config.optimization.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimization.grad_clip_norm
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
        if flush or bidx % self.config.schedule.log_every == 0:
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

    # ----- validation phase
    def _compute_iou(self) -> None:
        '''
        Finalize IoU metrics after validation batches.

        This method performs phase-level aggregation and formatting
        of metrics accumulated during batch execution. It deliberately
        lives at the Trainer/Evaluator layer rather than in EngineCore.
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
        '''
        Track best validation metrics and update patience counters.

        This method interprets finalized validation metrics according to
        tracking configuration and updates experiment-level monitoring
        state (best value, best epoch, patience counter).
        '''

        # get metric from validation metrics dictionary
        track_head = self.config.monitor.track_head_name
        val = self.state.epoch_sum.val_logs.head_metrics[track_head]
        met = val['ac_mean'] if val['has_active'] else val['mean']

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
        if self.config.monitor.track_mode == 'max':
            # update tracking numbers
            if met >= self.state.metrics.best_value + delta:
                self.state.metrics.best_value = met
                self.state.metrics.best_epoch = self.state.progress.epoch
                self.state.metrics.patience_n = 0 # reset patience
            # otherwise increment patience counter
            else:
                self.state.metrics.patience_n += 1
