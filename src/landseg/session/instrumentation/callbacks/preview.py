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

'''Preivew callback'''

# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.instrumentation.callbacks as callbacks
import landseg.session.instrumentation.formatters as formatters

class PreviewCallback(callbacks.BaseCallback):
    '''Preview callback.'''

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
        # collect stitched tensors into a dict
        stitched = {}
        # stitched = {
        #     'raw_image': formatters.stitch_patches(infer.infer_image)[0] # test - elevation
        # }
        # targets per head
        for head, targets in infer.infer_targets.items():
            stitched[f'{head}_labels'] = formatters.stitch_patches(
                targets,
                palette=self.reclass_color_map
            )
        # predictions per head
        for head, preds in infer.infer_preds.items():
            stitched[f'{head}_predictions'] = formatters.stitch_patches(
                preds,
                palette=self.reclass_color_map
            )
        # errors per head
        for head, errors in infer.infer_errors.items():
            stitched[f'{head}_errors'] = formatters.stitch_patches(
                errors,
                palette={1: [40, 40, 40], 0: [255, 140, 0]} # grey vs orange
            )
        # broadcast to trackers
        for tracker in self._trackers:
            for key, t in stitched.items():
                tracker.log_image(f'{phase}_{key}', t, step)
            tracker.flush()

    def on_train_phase_end(self, phase: str, reason: str): ...

    def on_train_end(self) -> None:
        for tracker in self._trackers:
            tracker.close()

    def on_checkpointing(self, fp: str): ...
