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
# pylint: disable=too-many-public-methods
# pylint: disable=unused-argument

'''
Defines the base callback interface and supporting state protocols
used by the training engine.

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

# ---------------------------------Public Type---------------------------------
class EngineStateLike(typing.Protocol):
    '''Interface on the subset of engine state for callbacks.'''
    progress: _Progress
    heads: _Heads
    batch_cxt: _BatchContex
    batch_out: _BatchOutput
    summary: _EpochSummary

class _Progress(typing.Protocol):
    epoch: int
    epoch_step: int
    global_step: int

class _Heads(typing.Protocol):
    all_heads: list[str]
    active_hmetrics: dict[str, _MetricsModule] | None

class _MetricsModule(typing.Protocol):
    def reset(self, device: str) -> None: ...

class _BatchContex(typing.Protocol):
    def refresh(self, bidx: int, batch: tuple) -> None: ...

class _BatchOutput(typing.Protocol):
    def refresh(self, bidx: int) -> None: ...

class _EpochSummary(typing.Protocol):
    train_summary: _TrainSummary
    val_summary: _ValSummary
    infer_context: _InferContext

class _TrainSummary(typing.Protocol):
    def clear(self) -> None: ...

class _ValSummary(typing.Protocol):
    def clear(self) -> None: ...

class _InferContext(typing.Protocol):
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]
    maps: dict

# --------------------------------Public  Class--------------------------------
class Callback:
    '''
    Base class for defining training engine callbacks.

    Subclasses can override any of the hook methods to execute custom
    logic at specific points in the training, validation, or inference
    lifecycle. All methods are optional; only override those needed.

    The callback operates on a shared `state` object that conforms to
    `EngineStateLike`, enabling interaction with training progress,
    batch data, metrics, and summaries without tight coupling to a
    specific engine implementation.
    '''

    def __init__(
        self,
        state: EngineStateLike,
        *,
        device: str,
        verbose: bool = True
    ):
        '''
        Initializes the callback.

        Args:
            state: Engine state object providing access to runtime
                context, including progress, batch data, and summaries.
            device: Target device identifier (e.g., "cpu", "cuda") used
                for any device-specific operations within the callback.
            verbose: If True, enables optional logging or debug output
                in callback implementations..
        '''
        self.state = state
        self.device = device
        self.verbose = verbose

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
