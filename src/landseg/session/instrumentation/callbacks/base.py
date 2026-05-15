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
# pylint: disable=too-few-public-methods
# pylint: disable=unused-argument

'''
Defines the base callback interface and supporting state protocols.

This module provides:
- A `Callback` base class that users can subclass to inject custom
  behavior at specific stages of training, validation, and inference.
- A set of `Protocol` definitions that describe the expected structure
  of the engine state object passed to callbacks.

The design relies on structural typing (via `typing.Protocol`) to keep
the engine loosely coupled while ensuring type safety and clarity of
expected interfaces.
'''

# standard imports
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.instrumentation.dashboards as dashboards

if typing.TYPE_CHECKING:
    import torch

# --------------------------------Public  Class--------------------------------
class BaseCallback(common.SessionObserverLike):
    '''
    Base class for defining dashboarding callbacks.

    Subclasses can override any of the hook methods to execute custom
    logic at specific points in the training, validation, or inference
    lifecycle. All methods are optional; only override those needed.
    '''

    def __init__(
        self,
        trackers: list[dashboards.BaseTracker] | None = None,
        *,
        verbose: bool = True,
        reclass_color_map: dict[int, list[int]] | None = None
    ):
        '''
        Initializes the callback.

        Args:
            verbose: If True, enables optional logging or debug output
                in callback implementations..
        '''
        self._trackers: list[dashboards.BaseTracker] = []
        if trackers:
            self._trackers = trackers
        self.verbose = verbose
        self._reclass_color_map = reclass_color_map
        self._infer_logs: dict[str, str] = {}
        self._infer_tensors: 'dict[str, torch.Tensor]' = {}

    # --- training phase begins
    def on_train_phase_begin(self, phase: common.PhaseLike) -> None: ...
    # --- training step begins
    def on_train_step_begin(self) -> None: ...
    # --- epoch begins
    def on_epoch_begin(self, epoch: int) -> None: ...
    # --- policy begins
    def on_train_policy_begin(self) -> None: ...
    def on_val_policy_begin(self) -> None: ...
    def on_infer_policy_begin(self) -> None: ...
    # --- batch begins
    def on_batch_begin(self, action: str, bidx: int) -> None: ...
    # --- batch ends
    def on_train_batch_end(self, bidx: int, results: core.TrainerEpochResults) -> None: ...
    def on_val_batch_end(self) -> None: ...
    def on_infer_batch_end(self) -> None: ...
    # --- policy ends
    def on_train_policy_end(self, results: core.TrainerEpochResults) -> None: ...
    def on_val_policy_end(self, results: core.ValidationEpochResults) -> None: ...
    def on_infer_policy_end(self, results: core.ValidationEpochResults) -> None: ...
    # --- epoch ends
    def on_epoch_end(self, epoch: int) -> None: ...
    # --- training step ends
    def on_train_step_end(self, results: core.TrainingSessionStep) -> None: ...
    # --- training phase ends
    def on_train_phase_end(self, phase: str, reason: str) -> None: ...
      # --- training end
    def on_train_end(self) -> None: ...
    # --- utilities
    def on_checkpointing(self, fp: str) -> None: ...
