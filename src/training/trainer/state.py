'''Module internal: trainer runtime state.'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# third-party imports
import torch
# local imports
import alias
import training.common

# ----- .progress
@dataclasses.dataclass
class _Progress:
    '''Trainig progress.'''
    epoch: int = 0
    epoch_step: int = 0
    global_step: int =  0

    def __str__(self) -> str:
        return '\n'.join([
            'Progress:',
            f'\tCurrent Epoch: {self.epoch}',
            f'\tCurrent Step in Epoch: {self.epoch_step}',
            f'\tCurrent Global Step: {self.global_step}'
        ])

# ----- .heads
@dataclasses.dataclass
class _Heads:
    '''Heads related state.'''
    all_heads: list[str] = dataclasses.field(init=False)
    active_heads: list[str] | None = None
    frozen_heads: list[str] | None = None
    active_hspecs: dict[str, training.common.SpecLike] | None = None
    active_hloss: dict[str, training.common.CompositeLossLike] | None = None
    active_hmetrics: dict[str, training.common.MetricLike] | None = None

    def __str__(self) -> str:
        return '\n'.join([
            'Head status:',
            f'\tActive Heads: {self.list_to_str(self.active_heads)}',
            f'\tFrozen Heads: {self.list_to_str(self.frozen_heads)}'
        ])

    @staticmethod
    def list_to_str(lst: list[str] | None) -> str:
        '''Turn list of strings to a joined string.'''
        if lst is None:
            return 'N/A'
        return '|'.join(lst)

# ----- .batch_ctx (context)
@dataclasses.dataclass
class _BatchCtx:
    '''Batch level data context.'''
    bidx: int = 0
    pidx_start: int = 0
    batch_size_full: int = 0
    batch: alias.DatasetItem | None = None
    x: torch.Tensor = torch.empty(0)
    y_dict: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    domain: dict[str, torch.Tensor | None] = dataclasses.field(default_factory=dict)

    def __str__(self):
        x = self.x.shape if self.x.numel() != 0 else 'N/A'
        return '\n'.join([
            'Batch Context',
            f'\tCurrent Batch ID: {self.bidx}',
            f'\tBatch X Dimension: {x}',
            f'\tBatch Y Head Counts: {len(self.y_dict)}',
            f'\tBatch Domain in Use: {self._active_domain()}'
        ])

    def _active_domain(self) -> str:
        '''Active domain'''
        out: list[str] = []
        for k, v in self.domain.items():
            if v is not None:
                out.append(f'{k}: {v.shape}')
        if out:
            return '|'.join(out)
        return 'N/A'

    def refresh(self, bidx: int, batch: tuple) -> None:
        '''Refresh context at the beginning of a batch.'''
        # take input from new batch
        self.bidx = bidx
        self.batch = batch
        # calc starting patch id of this batch
        self.pidx_start = (bidx - 1) * self.batch_size_full
        # clear old batch
        self.x = torch.empty(0)
        self.y_dict.clear()
        self.domain.clear()

# ----- .batch_out (output)
@dataclasses.dataclass
class _BatchOut:
    '''Batch level outputs.'''
    bidx: int = 0
    preds: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    total_loss: torch.Tensor = torch.empty(0)
    head_loss: dict[str, float] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        loss = self.total_loss.detach().item() \
            if self.total_loss.numel() != 0 else 'N/A'
        return '\n'.join([
            'Batch Output:',
            f'\tCurrent Batch ID: {self.bidx}',
            f'\tBatch Prediction Head Counts: {len(self.preds)}',
            f'\tBatch Total Loss: {loss}',
            f'\tBatch Per-head Loss: {self._perhead_loss()}'
        ])

    def _perhead_loss(self) -> str:
        '''Per-head loss.'''
        if self.head_loss:
            return '|'.join([f'{k}={v:.4f}' for k, v in self.head_loss.items()])
        return 'N/A'

    def refresh(self, bidx: int):
        '''Clean up for a new batch.'''
        self.bidx = bidx                            # take input from new batch
        self.preds.clear()                          # clear the old batch
        self.total_loss = torch.empty(0)            # clear the old batch
        self.head_loss.clear()                      # clear the old batch

# ----- .epoch_sum (summary)
@dataclasses.dataclass
class _TrainLogs:
    '''Training result summary.'''
    head_losses: dict[str, float] = dataclasses.field(default_factory=dict)
    head_losses_str: str = ''
    updated: bool = False

@dataclasses.dataclass
class _ValLogs:
    '''Validation result summary.'''
    head_metrics: dict[str, dict[str, typing.Any]] = dataclasses.field(default_factory=dict)
    head_metrics_str: dict[str, list[str]] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class _InferContext:
    '''Inference result summary.'''
    patch_per_blk: int = 0
    patch_per_dim: int = 0
    block_columns: int = 0
    patch_grid_shape: tuple[int, int] = (0, 0)
    maps: dict[str, dict[tuple[int, int], torch.Tensor]] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class _Epoch:
    '''Epoch level summary'''
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_logs: _TrainLogs = dataclasses.field(default_factory=_TrainLogs)
    val_logs: _ValLogs = dataclasses.field(default_factory=_ValLogs)
    infer_ctx: _InferContext = dataclasses.field(default_factory=_InferContext)

    def __str__(self) -> str:
        return '\n'.join([
            'Epoch Results:',
            f'\tTraining Loss: {self.train_loss}',
            f'\tValidation Loss: {self.val_loss}',
        ])

# ----- .metrics
@dataclasses.dataclass
class _Metrics:
    '''Experiment level metrics'''
    last_value: float = -float('inf')
    curr_value: float = -float('inf')
    best_value: float = -float('inf')
    best_epoch: int = -1
    patience_n: int = 0

    def __str__(self) -> str:
        _lastv = self.last_value if self.last_value != -float('inf') else 'N/A'
        _currv = self.curr_value if self.last_value != -float('inf') else 'N/A'
        _bestv = self.best_value if self.best_value != -float('inf') else 'N/A'
        _epoch = self.best_epoch if self.best_epoch != -1 else 'N/A'
        return '\n'.join([
            'Experimental Metrics:',
            f'\tLast Value: {_lastv}',
            f'\tCurr Value: {_currv}',
            f'\tBest Value: {_bestv}',
            f'\tBest Epoch: {_epoch}',
            f'\tPatience: {self.patience_n}'
        ])

# ----- .optim (optimization state)
@dataclasses.dataclass
class _Optim:
    '''Optimization state.'''
    scaler: torch.GradScaler = dataclasses.field(init=False)

    def __str__(self) -> str:
        if self.scaler is None:
            scaler_text = 'Scaler: Not Initiated'
        else:
            if self.scaler._enabled:
                scale = self.scaler.get_scale()
                scaler_text = f'Scaler: Enabled, Current Scale: {scale}'
            else:
                scaler_text = 'Scaler: Not Enabled'

        return '\n'.join([
            'Optimization Status:',
            f'\t{scaler_text}',
        ])

# ----- Runtime state (composite)
@dataclasses.dataclass
class RuntimeState:
    '''Training state with defualt values.'''
    progress: _Progress = dataclasses.field(default_factory=_Progress)
    heads: _Heads = dataclasses.field(default_factory=_Heads)
    batch_cxt: _BatchCtx = dataclasses.field(default_factory=_BatchCtx)
    batch_out: _BatchOut = dataclasses.field(default_factory=_BatchOut)
    epoch_sum: _Epoch = dataclasses.field(default_factory=_Epoch)
    metrics: _Metrics = dataclasses.field(default_factory=_Metrics)
    optim: _Optim = dataclasses.field(default_factory=_Optim)

    def __str__(self):
        return '\n'.join([
            f'{str(self.progress)}',
            f'{str(self.heads)}',
            f'{str(self.batch_cxt)}',
            f'{str(self.batch_out)}',
            f'{str(self.epoch_sum)}',
            f'{str(self.metrics)}',
            f'{str(self.optim)}'
        ])
