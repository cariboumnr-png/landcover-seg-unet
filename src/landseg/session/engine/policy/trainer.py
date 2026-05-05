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
        self.results = core.TrainerEpochResults(
            all_heads=self.state.heads.all_heads,
            current_lr=self.state.optim.lr
        )
        # epoch-level accumulated loss tracker
        self._loss: float
        self._head_losses: dict[str, float]

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

        Lifecycle callback hooks are emitted as semantic markers to allow
        observation and side effects, but do not invoke or control
        execution logic.

        Returns:
            A dictionary of averaged epoch-level training metrics
            (e.g., total and per-head losses).
        '''

        # training phase begin
        self.dispatcher.on_train_policy_begin()
        # set model to train mode
        self.model.train()
        # reset results container (avoid carry-over from last epoch)
        self.results.clear()
        # update epoch tracker
        self.state.progress.epoch = epoch
        # reset loss trackers
        self._loss = 0.0
        self._head_losses = {h: 0.0 for h in self.state.heads.all_heads}

        # interate through training data batches
        assert self.dataloaders.train, 'Training dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.train, start=1):

            # batch start
            self.dispatcher.on_batch_begin('Training', bidx)
            self._batch_reset(bidx, batch)

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
            # add to interal loss trackers
            self._loss += total.detach().item()
            for head, loss in self.state.batch_out.head_loss.items():
                self._head_losses[head] += loss

            # gradient clipping
            optimizer = self.optimization.optimizer
            # unscale if use AMP
            if self.engine.config.use_amp:
                self.state.optim.scaler.unscale_(optimizer)
            self._clip_grad()

            # optimizer step
            optimizer = self.optimization.optimizer
            # use AMP
            if self.engine.config.use_amp:
                self.state.optim.scaler.step(optimizer)
                self.state.optim.scaler.update() # update scaler
            # no AMP
            else:
                self.optimization.optimizer.step()

            # scheduler step (if present)
            if self.optimization.scheduler is not None:
                self.optimization.scheduler.step()

            # increment global step counter
            self.state.progress.global_step += 1
            # snapshopt learning rate
            self.state.optim.lrs = [g['lr'] for g in optimizer.param_groups]

            # batch end
            self._update_training_stats() # depending on frequency config
            self.dispatcher.on_train_batch_end(self.results)

        # training phase end
        self._update_training_stats(flush=True) # force update
        self.dispatcher.on_train_policy_end(self.results)
        return self.results

    # ----- training phase
    def _clip_grad(self):
        '''Clip gradients by global norm when set.'''

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm
            )

    def _update_training_stats(self, flush: bool=False):
        '''
        Average losses and update per-head loss logs at intervals.

        Set `flush=True` to flush results at end of a training epoch.
        '''

        # get current epoch batch id
        bidx = self.state.batch_cxt.bidx
        n = max(1, bidx) # safe denominator
        # update results container
        if flush or bidx % self.update_every == 0:
            # --- total loss
            self.results.total_loss = self._loss / n
            # --- perhead loss
            for head in self.state.heads.all_heads:
                self.results.head_losses[head] = self._head_losses[head] / n
            # --- global step
            self.results.last_updated = self.state.progress.global_step
            # --- current learning rate
            self.results.current_lr = self.state.optim.lr

            # # pretty string of the losses
            # logs = {}
            # logs['total'] = self.results.total_loss
            # logs = {h: l for h, l in self.results.head_losses.items() if l > 0}
            # text_list = [f'{k}: {v:.4f}' for k, v in logs.items()]
            # text = f'batch_{bidx:04d} | ' + '|'.join(text_list)
            # print(text)
