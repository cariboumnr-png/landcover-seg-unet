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

# local imports
import landseg.session.engine as engine

class MultiHeadEvaluator(engine.EngineBase):
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

    def __init__(self,
        *,
        track_mode: str,
        track_head_name: str,
        min_delta: float | None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.track_mode = track_mode
        self.track_head_name = track_head_name
        self.min_delta = min_delta

    # -------------------------------Public  Methods-------------------------------
    def validate(self) -> dict[str, dict]:
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
        self._compute_iou()
        self._track_metrics()
        self._emit('on_validation_end')
        return self.state.epoch_sum.val_logs.head_metrics

    def infer(self, out_dir: str) -> None:
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
        self._emit('on_inference_end', out_dir)

    # ----- validation phase
    def _compute_iou(self) -> None:
        '''
        Finalize IoU and related validation metrics.

        This method performs phase-level aggregation and formatting of
        metrics accumulated during validation batches. It deliberately
        lives at the evaluator policy layer rather than in the execution
        core, which remains metric-agnostic.
        '''

        assert self.state.heads.active_hmetrics is not None
        val_logs: dict[str, dict] = {}
        val_logs_text: dict[str, list[str]] = {}

        for head, metrics_module in self.state.heads.active_hmetrics.items():
            metrics_module.compute()
            val_logs[head] = metrics_module.metrics_dict
            val_logs_text[head] = metrics_module.metrics_text

        self.state.epoch_sum.val_logs.head_metrics = val_logs
        self.state.epoch_sum.val_logs.head_metrics_str = val_logs_text

    def _track_metrics(self) -> None:
        '''
        Track best validation metrics and update patience counters.

        This method interprets finalized validation metrics according to
        monitoring configuration and updates experiment-level tracking
        state, including:

        - best metric value
        - best epoch
        - patience counter
        '''

        # get metric from validation metrics dictionary
        track_head = self.track_head_name
        val = self.state.epoch_sum.val_logs.head_metrics[track_head]
        met = val['ac_mean'] if val['has_active'] else val['mean']

        # at the end of the first epoch
        if self.state.progress.epoch == 1:
            self.state.metrics.last_value = 0.0
            self.state.metrics.curr_value = met
        else:
            self.state.metrics.last_value = self.state.metrics.curr_value
            self.state.metrics.curr_value = met

        delta = self.min_delta or 0.0
        assert delta >= 0.0

        if self.track_mode == 'max':
            # update tracking numbers
            if met >= self.state.metrics.best_value + delta:
                self.state.metrics.best_value = met
                self.state.metrics.best_epoch = self.state.progress.epoch
                self.state.metrics.patience_n = 0
            else:
                self.state.metrics.patience_n += 1
