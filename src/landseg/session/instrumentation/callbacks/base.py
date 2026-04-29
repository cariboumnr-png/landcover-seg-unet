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

# pylint: disable=missing-function-docstring
# pylint: disable=too-many-public-methods
# pylint: disable=unused-argument

'''Base class for trainer callbacks.'''

# local imports
import landseg.session.common as common

class Callback:
    '''Base class for callbacks; subclass to implement behaviors.'''

    def __init__(self):
        self._state: common.StateLike | None = None
        self._config: common.ConfigLike | None = None
        self._device: str | None
        self.verbose: bool = True

    def setup(
        self,
        state: common.StateLike,
        config: common.ConfigLike,
        *,
        device: str,
    ) -> None:
        self._state = state
        self._config = config
        self._device = device

    # -----------------------------training phase-----------------------------
    def on_train_epoch_begin(self, epoch: int) -> None: ...
    def on_train_batch_begin(self, bidx: int, batch: tuple) -> None: ...
    def on_train_batch_forward(self) -> None: ...
    def on_train_batch_compute_loss(self) -> None: ...
    def on_train_backward(self) -> None: ...
    def on_train_before_optimizer_step(self) -> None: ...
    def on_train_optimizer_step(self) -> None: ...
    def on_train_batch_end(self) -> None: ...
    def on_train_epoch_end(self) -> None: ...

    # ----------------------------validation phase----------------------------
    def on_validation_begin(self) -> None: ...
    def on_validation_batch_begin(self, bidx: int, batch: tuple) -> None: ...
    def on_validation_batch_forward(self) -> None: ...
    def on_validation_batch_end(self) -> None: ...
    def on_validation_end(self) -> None: ...

    # -----------------------------inference phase-----------------------------
    def on_inference_begin(self) -> None: ...
    def on_inference_batch_begin(self, bidx: int, batch: tuple) -> None: ...
    def on_inference_batch_forward(self) -> None: ...
    def on_inference_batch_end(self) -> None: ...
    def on_inference_end(self, out_dir: str, **kwargs) -> None: ...

    # -------------------------convenience properties-------------------------
    @property
    def state(self) -> common.StateLike:
        if self._state is None:
            raise RuntimeError('Runtime State accessed before setup.')
        return self._state

    @property
    def config(self) -> common.ConfigLike:
        if self._config is None:
            raise RuntimeError('Engine config accessed before setup.')
        return self._config\

    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError('Engine config accessed before setup.')
        return self._device
