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
Multi-head evaluation policy for hierarchical segmentation.

This module defines `MultiHeadEvaluator`, which provides epoch-level
validation and inference policies on top of a shared batch execution
engine.

The evaluator is responsible for:
- Defining validation and inference workflows
- Controlling evaluation-time head configuration
- Aggregating and formatting epoch-level metrics
- Tracking best validation metrics and patience state
- Emitting lifecycle callbacks as semantic markers

The evaluator deliberately does NOT:
- Perform batch-level execution (forward, loss, metrics)
- Parse input batches
- Compute losses or metrics incrementally
- Manage optimizer or gradient state

Those responsibilities are delegated to `BatchExecutionEngine`, which
mutates shared runtime state during batch execution.

In short:
- The batch executor answers: "What happened in this batch?"
- The evaluator answers: "How should batch results be interpreted?"
'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.engine.policy as policy

class MultiHeadEvaluator(policy.EngineBase):
    '''
    Evaluation and inference policy controller.

    This class defines epoch-level behavior for validation and inference
    by orchestrating:

    - Batch execution via a shared BatchExecutionEngine
    - Evaluation-time head activation and exclusions
    - Epoch-level metric aggregation and finalization
    - Best-metric tracking and patience updates

    The evaluator operates on a shared RuntimeState object that persists
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
        monitor_heads: list[str],
        dataset: typing.Literal['val', 'test'] = 'val',
        val_every: int = 1,
        infer_every: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.monitor_heads = monitor_heads
        self.dataset: typing.Literal['val', 'test'] = dataset
        self.val_every = val_every
        self.infer_every = infer_every

        # init the epoch results container with all heads
        self.results = core.EvaluatorEpochResults(
            all_heads=self.state.heads.all_heads,
            monitor_heads=self.monitor_heads
        )

    # -------------------------------Public  Methods-------------------------------
    def validate(self, epoch: int) -> core.EvaluatorEpochResults | None:
        '''
        Execute a full validation epoch and return finalized metrics.

        This method defines validation policy, including:

        - Switching the model to evaluation mode
        - Configuring validation-time logit adjustment
        - Delegating batch execution to the execution engine
        - Finalizing and formatting epoch-level metrics
        - Updating best-metric tracking and patience state

        Batch-level metric accumulation (e.g., confusion matrices) is
        performed by the batch execution engine during validation batches.
        Epoch-level computation and interpretation are performed here.

        Lifecycle callback hooks are emitted as semantic markers for
        observation and side effects.

        Returns:
            A mapping from head name to its finalized validation metrics.
        '''

        # early exit if this epoch is not to be validated
        if not epoch % self.val_every == 0:
            return None

        # validation phase begin
        self._emit('on_val_policy_begin')
        # set model to evaluation mode
        self.model.eval()
        # reset per-head confusion matrix from active heads
        assert self.state.heads.active_hmetrics is not None
        for metrics_mod in self.state.heads.active_hmetrics.values():
            metrics_mod.reset(self.device)
        # reset head metrics dictionary
        self.results.head_metrics.clear()

        # set target dataset
        match self.dataset:
            case 'val': dataloader = self.dataloaders.val
            case 'test': dataloader = self.dataloaders.test

        # iterate through validation dataset
        assert dataloader, 'Target dataset not provided'
        for bidx, batch in enumerate(dataloader, start=1):

            # batch start
            self._emit('on_batch_begin', 'Validating', bidx)
            self._batch_reset(bidx, batch)

            # delegate to batch executor
            self.engine.run_validate_batch()
            self._emit('val_batch_end')

        # val phase end
        self._compute_iou()
        self._emit('val_policy_end')
        return self.results

    def infer(self, epoch: int) -> None:
        '''
        Execute inference over the test dataset.

        This method defines inference policy, including:

        - Switching the model to evaluation mode
        - Configuring inference-time logit adjustment
        - Delegating batch execution to the execution engine
        - Emitting inference lifecycle callbacks for side effects

        Batch-level inference execution and aggregation are handled by
        the batch execution engine. This method does not interpret or
        post-process inference results directly.
        '''

        # early exit if this epoch is not to be validated
        if not epoch % self.infer_every == 0:
            return

        # infer phase begin
        self._emit('on_infer_policy_begin')
        # set model to evaluation mode
        self.model.eval()

        # iterate through inference dataset
        assert self.dataloaders.test, 'Inference dataset not provided'
        for bidx, batch in enumerate(self.dataloaders.test, start=1):

            # batch start
            self._emit('on_batch_begin', 'Inferring', bidx)
            self._batch_reset(bidx, batch)

            # delegate to batch executor
            self.engine.run_infer_batch()
            self._emit('infer_batch_end')

        # inference phase end
        self.results.head_inference = self.state.batch_out.infer_maps
        self._emit('on_infer_policy_end')

    # ----- validation phase
    def _compute_iou(self) -> None:
        '''
        Finalize IoU and related validation metrics.

        This method performs phase-level aggregation and formatting of
        metrics accumulated during validation batches. It deliberately
        lives at the evaluator policy layer rather than in the execution
        core, which remains metric-agnostic.
        '''

        # sanity
        assert self.state.heads.active_hmetrics is not None

        for head, metrics_module in self.state.heads.active_hmetrics.items():
            # compute assign metrics to epoch results
            metrics_module.compute()
            self.results.head_metrics[head] = metrics_module.metrics
            # collect per head metrics formatted strings
            # metrics_str[head] = metrics_module.metrics.as_str_list
