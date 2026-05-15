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
        metrics = results.metrics
        phase = results.phase_name
        step = results.epoch_in_phase
        if not metrics.inference:
            return
        infer = metrics.inference
        # all should have the same heads but here we will use heads in targets
        for head in infer.infer_targets.keys():
            # targets
            targets = formatters.stitch_patches(infer.infer_targets[head])
            # predictions
            preds = formatters.stitch_patches(infer.infer_preds[head])
            # errors
            errors = formatters.stitch_patches(infer.infer_errors[head])

            # assign to tags
            # add once
            if f'{head}_labels' not in self._infer_logs:
                self._infer_logs[f'{head}_labels'] = formatters.colorize(
                    targets,
                    palette=self._reclass_color_map
                )
            # refresh every call
            self._infer_logs[f'{head}_predictions'] = formatters.colorize(
                preds,
                palette=self._reclass_color_map
            )
            self._infer_logs[f'{head}_errors'] = formatters.colorize(
                errors,
                palette={1: [40, 40, 40], 0: [255, 140, 0]} # grey vs orange
            )

            # from confusion matrics
            _, cm_text = formatters.get_cmatrix(
                targets,
                preds,
                class_range=(1, 6),
                class_names=['WAT', 'FOR', 'WET', 'NT', 'DISTB', 'OTH'],
                exclude_cls=(4, 6),
                ignore_index=255,
            )
            print(cm_text)

        # broadcast to trackers
        for tracker in self._trackers:
            for key, t in self._infer_logs.items():
                if t.dim() == 3:
                    tracker.log_image(f'{phase}_{key}', t, step)
                elif t.dim() == 2:
                    tracker.log_image(f'{phase}_{key}', t, step, dataformats='HW')
            tracker.flush()

    def on_train_phase_end(self, phase: str, reason: str): ...

    def on_train_end(self) -> None:
        for tracker in self._trackers:
            tracker.close()

    def on_checkpointing(self, fp: str): ...
