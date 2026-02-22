# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''
Callback facing trainer runtime state protocols.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import training.common

if typing.TYPE_CHECKING:
    import torch

# ----- Runtime state (composite)
@typing.runtime_checkable
class RuntimeStateLike(typing.Protocol):
    progress: Progress
    heads: Heads
    batch_cxt: BatchCtx
    batch_out: BatchOut
    epoch_sum: Epoch
    metrics: Metrics
    optim: OptimState

# ----- .progress
class Progress(typing.Protocol):
    epoch: int
    epoch_step: int
    global_step: int

# ----- .heads
class Heads(typing.Protocol):
    all_heads: list[str]
    active_heads: list[str] | None
    frozen_heads: list[str] | None
    active_hspecs: dict[str, training.common.SpecLike] | None
    active_hloss: dict[str, training.common.CompositeLossLike] | None
    active_hmetrics: dict[str, training.common.MetricLike] | None

# ----- .batch_ctx (context)
class BatchCtx(typing.Protocol):
    bidx: int
    block_index_range: tuple[int, int]
    batch: tuple['torch.Tensor', dict, dict] | None
    x: 'torch.Tensor'
    y_dict: dict[str, 'torch.Tensor']
    domain: dict[str, 'torch.Tensor| None']
    def refresh(self, bidx: int, batch: tuple) -> None: ...

# ----- .batch_out (output)
class BatchOut(typing.Protocol):
    bdix: int
    preds: dict[str, 'torch.Tensor']
    total_loss: 'torch.Tensor'
    head_loss: dict[str, float]
    def refresh(self, bidx) -> None: ...

# ----- .epoch_sum (summary)
class Epoch(typing.Protocol):
    train_loss: float
    val_loss: float
    train_logs: _TrainLogs
    val_logs: _ValLogs
    infer_output: _InferOutputs

class _TrainLogs(typing.Protocol):
    '''Training logs.'''
    head_losses: dict[str, float]
    head_losses_str: str
    updated: bool

class _ValLogs(typing.Protocol):
    '''Validation logs.'''
    head_metrics: dict[str, dict[str, typing.Any]]
    head_metrics_str: dict[str, list[str]]

class _InferOutputs(typing.Protocol):
    '''Inference logs.'''
    maps: dict[str, dict[tuple[int, int], torch.Tensor]]

# ----- .metrics
class Metrics(typing.Protocol):
    last_value: float
    curr_value: float
    best_value: float
    best_epoch: int
    patience_n: int

# ----- .optim (optimization state)
class OptimState(typing.Protocol):
    scaler: 'torch.GradScaler'
