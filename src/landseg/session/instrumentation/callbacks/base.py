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
from __future__ import annotations
import typing

# imports only when type checking
if typing.TYPE_CHECKING:
    import torch

# ---------------------------------Public Type---------------------------------
class EngineStateLike(typing.Protocol):
    '''Interface of engine state that the callbacks listen to.'''
    progress: _Progress
    optim: _OptimeRuntime

class _Progress(typing.Protocol):
    @property
    def epoch(self) -> int: ...
    @property
    def global_step(self) -> int: ...

class _OptimeRuntime(typing.Protocol):
    @property
    def lr(self) -> float: ...

class SessionStepLike(typing.Protocol):
    '''Interface of epoch results that the callbacks listen to'''
    training: _TrainerEpochResults | None
    evaluation: _EvaluatorEpochResults | None

class _TrainerEpochResults(typing.Protocol):
    @property
    def total_loss(self) -> float: ...
    @property
    def head_loss(self) -> dict[str, float]: ...

class _EvaluatorEpochResults(typing.Protocol):
    @property
    def target_metrics(self) -> float: ...
    @property
    def head_inference(self) -> dict[str, dict[tuple[int, int], 'torch.Tensor']]: ...
# --------------------------------Public  Class--------------------------------
class BaseCallback:
    '''
    Base class for defining dashboarding callbacks.

    Subclasses can override any of the hook methods to execute custom
    logic at specific points in the training, validation, or inference
    lifecycle. All methods are optional; only override those needed.
    '''

    def __init__(self, *, verbose: bool = True):
        '''
        Initializes the callback.

        Args:
            verbose: If True, enables optional logging or debug output
                in callback implementations..
        '''
        self.verbose = verbose

    # base methods
    # --- epoch begins
    def on_epoch_begin(self, epoch: int) -> None: ...
    # --- policy begins
    def on_train_policy_begin(self) -> None: ...
    def on_val_policy_begin(self) -> None: ...
    def on_infer_policy_begin(self) -> None: ...
    # --- batch begins
    def on_batch_begin(self, action: str, bidx: int) -> None: ...
    # --- batch ends
    def on_train_batch_end(self) -> None: ...
    def on_val_batch_end(self) -> None: ...
    def on_infer_batch_end(self) -> None: ...
    # --- policy ends
    def on_train_policy_end(self) -> None: ...
    def on_val_policy_end(self) -> None: ...
    def on_infer_policy_end(self) -> None: ...
    # --- epoch ends
    def on_epoch_end(self, epoch: int) -> None: ...
