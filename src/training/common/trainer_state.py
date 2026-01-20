# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''
Trainer runtime state protocols.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import training.common

if typing.TYPE_CHECKING:
    import torch

# -------------------------------trainer state-------------------------------
@typing.runtime_checkable
class RuntimeStateLike(typing.Protocol):
    progress: Progress
    heads: Heads
    batch_cxt: BatchCtx
    batch_out: BatchOut
    epoch_sum: Epoch
    metrics: Metrics
    optim: OptimState

class Progress(typing.Protocol):
    epoch: int
    epoch_step: int
    global_step: int

class Heads(typing.Protocol):
    all_heads: list[str]
    active_heads: list[str] | None
    frozen_heads: list[str] | None
    active_hspecs: dict[str, training.common.SpecLike] | None
    active_hloss: dict[str, training.common.CompositeLossLike] | None
    active_hmetrics: dict[str, training.common.MetricLike] | None

class BatchCtx(typing.Protocol):
    bidx: int
    batch: tuple['torch.Tensor', dict, dict] | None
    x: 'torch.Tensor'
    y_dict: dict[str, 'torch.Tensor']
    domain: dict[str, 'torch.Tensor| None']
    def refresh(self, bidx: int, batch: tuple) -> None: ...

class BatchOut(typing.Protocol):
    bdix: int
    preds: dict[str, 'torch.Tensor']
    total_loss: 'torch.Tensor'
    head_loss: dict[str, float]
    def refresh(self, bidx) -> None: ...

class Epoch(typing.Protocol):
    train_loss: float
    val_loss: float
    train_logs: dict[str, float]
    train_logs_text: str
    val_logs: dict[str, dict]
    val_logs_text: dict[str, list[str]]

class Metrics(typing.Protocol):
    last_value: float | None
    best_value: float
    best_epoch: int

class OptimState(typing.Protocol):
    scaler: 'torch.GradScaler'
