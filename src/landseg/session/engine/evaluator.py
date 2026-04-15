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

# local imports
import landseg.session.engine as engine

class MultiHeadEvaluator(engine.EngineBase):
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

# -------------------------------Public  Methods-------------------------------
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
