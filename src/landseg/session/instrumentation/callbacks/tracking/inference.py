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

'''Image callback'''

# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.instrumentation.callbacks as callbacks
import landseg.session.instrumentation.formatters as formatters

class InferTrackingCallback(callbacks.BaseCallback):
    '''Image callback.'''

    def on_train_phase_begin(self, phase: common.PhaseLike): ...

    def on_batch_begin(self, action: str, bidx: int): ...

    def on_train_batch_end(self, bidx: int, results: core.TrainerEpochResults): ...

    def on_train_step_end(self, results: core.TrainingSessionStep) -> None:
        # early exit if inference was not run
        infer_results = results.metrics.inference
        if not infer_results:
            return

        # all should have the same heads but here we will use heads in labels
        head_tensors = {}
        head_metrics = {}
        for head in infer_results.infer_labels.keys():
            # add tensors from label, preds, and errors
            head_tensors[f'{head}_labels'] = formatters.colorize(
                infer_results.infer_labels[head],
                palette=self._reclass_color_map
            )
            head_tensors[f'{head}_predictions'] = formatters.colorize(
                infer_results.infer_preds[head],
                palette=self._reclass_color_map
            )
            head_tensors[f'{head}_errors'] = formatters.colorize(
                infer_results.infer_errors[head],
                palette={1: [40, 40, 40], 0: [255, 140, 0]} # grey vs orange
            )
            # add mean IoU scalar
            head_metrics[head] = results.metrics.inference_metrics

        # broadcast to trackers
        phase = results.phase_name
        step = results.epoch_in_phase
        for tracker in self._trackers:
            # log images
            for k, v in head_tensors.items():
                if v.dim() == 3:
                    tracker.log_image(f'{phase}_{k}', v, step)
                elif v.dim() == 2:
                    tracker.log_image(f'{phase}_{k}', v, step, dataformats='HW')
            # log scalar
            for k, v in head_metrics:
                tracker.log_scalar(k, v, step)
            tracker.flush()

    def on_train_phase_end(self, phase: str, reason: str): ...

    def on_train_end(self) -> None:
        for tracker in self._trackers:
            tracker.close()

    def on_checkpointing(self, fp: str): ...
