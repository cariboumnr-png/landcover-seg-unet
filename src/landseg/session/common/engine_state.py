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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''
Callback-facing trainer runtime state protocols.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.session.common.engine_comps as engine_comps

if typing.TYPE_CHECKING:
    import torch

# ----- Runtime state (composite)
@typing.runtime_checkable
class StateLike(typing.Protocol):
    progress: _Progress
    heads: _Heads
    batch_cxt: _BatchCtx
    batch_out: _BatchOut
    epoch_sum: _Epoch
    metrics: _Metrics
    optim: _OptimState

# ----- .progress
class _Progress(typing.Protocol):
    epoch: int
    epoch_step: int
    global_step: int

# ----- .heads
class _Heads(typing.Protocol):
    all_heads: list[str]
    active_heads: list[str] | None
    frozen_heads: list[str] | None
    active_hspecs: dict[str, engine_comps.SpecsLike] | None
    active_hloss: dict[str, engine_comps.CompositeLossLike] | None
    active_hmetrics: dict[str, engine_comps.ConfusionMatrixLike] | None

# ----- .batch_ctx (context)
class _BatchCtx(typing.Protocol):
    bidx: int
    pidx_start: int
    batch: tuple['torch.Tensor', dict, dict] | None
    batch_size_full: int
    x: 'torch.Tensor'
    y_dict: dict[str, 'torch.Tensor']
    domain: dict[str, 'torch.Tensor| None']
    def refresh(self, bidx: int, batch: tuple) -> None: ...

# ----- .batch_out (output)
class _BatchOut(typing.Protocol):
    bdix: int
    preds: dict[str, 'torch.Tensor']
    total_loss: 'torch.Tensor'
    head_loss: dict[str, float]
    def refresh(self, bidx) -> None: ...

# ----- .epoch_sum (summary)
class _Epoch(typing.Protocol):
    train_loss: float
    val_loss: float
    train_logs: _TrainLogs
    val_logs: _ValLogs
    infer_ctx: _InferContext

class _TrainLogs(typing.Protocol):
    head_losses: dict[str, float]
    head_losses_str: str
    updated: bool

class _ValLogs(typing.Protocol):
    head_metrics: dict[str, dict[str, typing.Any]]
    head_metrics_str: dict[str, list[str]]

class _InferContext(typing.Protocol):
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]
    maps: dict[str, dict[tuple[int, int], torch.Tensor]]

# ----- .metrics
class _Metrics(typing.Protocol):
    last_value: float
    curr_value: float
    best_value: float
    best_epoch: int
    patience_n: int

# ----- .optim (optimization state)
class _OptimState(typing.Protocol):
    scaler: 'torch.GradScaler'
