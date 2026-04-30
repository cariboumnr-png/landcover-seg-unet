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

# third-party imports
import torch
# local imports
import landseg.core as core
import landseg.session.engine.policy as policy

class MultiHeadTrainer(policy.EngineBase):
    '''
    Training and evaluation policy controller.

    This class defines epoch-level behavior for training, validation,
    and inference by orchestrating:

    - Batch execution via a shared BatchExecutionEngine
    - Optimizer and scheduler behavior
    - Gradient scaling and clipping
    - Epoch-level loss aggregation

    The trainer operates on a shared RuntimeState object that persists
    across Runner, Trainer/Evaluator, and the batch execution engine.

    Batch-level mathematical execution is delegated entirely to
    BatchExecutionEngine. This class focuses exclusively on policy,
    orchestration, and interpretation of execution results.

    Lifecycle callback hooks emitted by this class serve as semantic
    markers for observation and side effects (logging, monitoring,
    visualization), but do not invoke or control execution logic.
    '''

    def __init__(
        self,
        *,
        grad_clip_norm: float | None,
        update_every: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grad_clip_norm = grad_clip_norm
        self.update_every = update_every

        # init the epoch results container with all heads
        self.results = core.TrainerEpochResults(self.state.heads.all_heads)

    def train_one_epoch(self, epoch: int) -> core.TrainerEpochResults:
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
        # reset results container (avoid carry-over from last epoch)
        self.results.clear()
        self._emit('on_train_epoch_begin', epoch)

        # interate through training data batches
        assert self.dataloaders.train, 'Training dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.train, start=1):

            # batch start
            self._emit('on_train_batch_begin', bidx, batch)
            # reset optimizer gradient
            self.optimization.optimizer.zero_grad(set_to_none=True)

            # delegate batch to engine (forward and compute loss)
            self.engine.run_train_batch()

            # batch backward on total loss
            total = self.state.batch_out.total_loss
            if self.engine.config.use_amp:
                self.state.optim.scaler.scale(total).backward()
            else:
                total.backward()
            self._emit('on_train_backward')

            # gradient clipping
            optimizer = self.optimization.optimizer
            # unscale if use AMP
            if self.engine.config.use_amp:
                self.state.optim.scaler.unscale_(optimizer)
            self._clip_grad()
            self._emit('on_train_before_optimizer_step')

            # optimizer step
            optimizer = self.optimization.optimizer
            # use AMP
            if self.engine.config.use_amp:
                self.state.optim.scaler.step(optimizer)
                self.state.optim.scaler.update() # update scaler
            # no AMP
            else:
                self.optimization.optimizer.step()
            self._emit('on_train_optimizer_step')

            # batch end
            # accumulate total loss
            batch_loss = float(self.state.batch_out.total_loss.detach().item())
            self.results.total_loss += batch_loss
            # accumulate per head loss
            for head, loss in self.state.batch_out.head_loss.items():
                if head in self.results.all_heads:
                    self.results.head_losses[head] += loss
            # update bidx
            self.results.current_bidx = bidx

            # update train logs if at interval (decided by trainer method)
            self._update_training_state(flush=False)
            self._emit('on_train_batch_end')

        # train phase end
        # - update logs and loss (total/per-head) for the epoch
        self._update_training_state(flush=True)
        self._emit('on_train_epoch_end')
        return self.results

    # ----- training phase
    def _clip_grad(self):
        '''Clip gradients by global norm when set.'''

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm
            )

    def _update_training_state(self, flush: bool=False):
        '''
        Average losses and update per-head loss logs at intervals.

        Set `flush=True` to flush results at end of a training epoch.
        '''

        # get current batch id
        bidx = self.state.batch_cxt.bidx
        # create log dict
        logs = {}
        # update log at interval
        if flush or bidx % self.update_every == 0:
            logs['Total_Loss'] = self.results.mean_total_loss
            # per-head losses
            for k, v in self.results.mean_head_losses.items():
                if v > 0: # only report non-zero losses
                    logs[f'Head_Loss_{k}'] = float(v)
            # pretty string of the losses
            text_list = [f'{k}: {v:.4f}' for k, v in logs.items()]
            text = f'batch_{bidx:04d} | ' + '|'.join(text_list)
            self.state.epoch.train_stats.total_loss = logs['Total_Loss']
            self.state.epoch.train_stats.head_losses_str = text
            self.state.epoch.train_stats.updated = True
        else:
            self.state.epoch.train_stats.updated = False # reset flag
