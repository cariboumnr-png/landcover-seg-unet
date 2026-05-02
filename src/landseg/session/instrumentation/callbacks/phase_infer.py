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

# pylint: disable=protected-access

'''Inference phase callback class.'''

# local imports
import landseg.session.instrumentation.callbacks as callbacks

class InferCallback(callbacks.Callback):
    '''Inference: parse -> forward -> collect outputs (optional).'''

    def on_inference_begin(self) -> None: ...

    def on_inference_batch_begin(self, bidx: int, batch: tuple) -> None: ...

    def on_inference_batch_forward(self) -> None: ...

    def on_inference_batch_end(self) -> None: ...

    def on_inference_end(self, out_dir: str, **kwargs) -> None: ...

        # ------------------------TO BE RE-IMPLEMENTED------------------------
        # # stitch all blocks together and output previews
        # # only if the patch grid is of valid shape, e.e, non-zero dims

        # # determine which heads to produce preview images
        # heads: list[str] = kwargs.get('preview_heads', [])
        # patch_grid_shape: tuple[int, int] = kwargs.get('patch_grid_shape', ())
        # # if no specifics provided, preview all heads
        # if not heads:
        #     heads = self.state.heads.all_heads
        # if all(patch_grid_shape):
        #     exporters.export_previews(
        #         self.state.epoch.eval_stats.infer_maps,
        #         out_dir,
        #         map_grid_shape=patch_grid_shape,
        #         heads=heads
        #     )
