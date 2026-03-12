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

'''Internal: trainer runtime state containers and pretty printers.'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# third-party imports
import torch
# local imports
import landseg.alias as alias
import landseg.core.trainer_protocols as trainer_protocols

# ------------------------------Public  Dataclass------------------------------
# ----- Runtime state (composite)
@dataclasses.dataclass
class RuntimeState:
    '''Composite training state with sensible defaults.'''
    progress: _Progress
    heads: _Heads
    batch_cxt: _BatchContex
    batch_out: _BatchOutput
    epoch_sum: _EpochSummary
    metrics: _MetricsTracker
    optim: _OptimState

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

# ------------------------------private dataclass------------------------------
# ----- .progress
@dataclasses.dataclass
class _Progress:
    '''Training progress counters (epoch/step/global).'''
    epoch: int
    epoch_step: int
    global_step: int

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
    '''State for multihead selection, freezing, and active specs.'''
    all_heads: list[str]
    active_heads: list[str] | None
    frozen_heads: list[str] | None
    active_hspecs: dict[str, trainer_protocols.SpecsLike] | None
    active_hloss: dict[str, trainer_protocols.CompositeLossLike] | None
    active_hmetrics: dict[str, trainer_protocols.ConfusionMatrixLike] | None

    def __str__(self) -> str:
        return '\n'.join([
            'Head status:',
            f'\tActive Heads: {self.list_to_str(self.active_heads)}',
            f'\tFrozen Heads: {self.list_to_str(self.frozen_heads)}'
        ])

    @staticmethod
    def list_to_str(lst: list[str] | None) -> str:
        '''Join a list of head names or return 'N/A' if None.'''
        if lst is None:
            return 'N/A'
        return '|'.join(lst)

# ----- .batch context
@dataclasses.dataclass
class _BatchContex:
    '''Per-batch input/context (indices, tensors, and domain info).'''
    bidx: int
    pidx_start: int
    batch_size_full: int
    batch: alias.DatasetItem | None
    x: torch.Tensor
    y_dict: dict[str, torch.Tensor]
    domain: dict[str, torch.Tensor | None]

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
        '''Summarize present domain tensors and their shapes.'''
        out: list[str] = []
        for k, v in self.domain.items():
            if v is not None:
                out.append(f'{k}: {v.shape}')
        if out:
            return '|'.join(out)
        return 'N/A'

    def refresh(self, bidx: int, batch: tuple) -> None:
        '''Reset batch context for a new iteration.'''
        # take input from new batch
        self.bidx = bidx
        self.batch = batch
        # calc starting patch id of this batch
        self.pidx_start = (bidx - 1) * self.batch_size_full
        # clear old batch
        self.x = torch.empty(0)
        self.y_dict.clear()
        self.domain.clear()

# ----- .batch output
@dataclasses.dataclass
class _BatchOutput:
    '''Per-batch outputs: predictions and losses.'''
    bidx: int
    preds: dict[str, torch.Tensor]
    total_loss: torch.Tensor
    head_loss: dict[str, float]

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
        '''Format per-head loss values or return 'N/A'.'''
        if self.head_loss:
            return '|'.join([f'{k}={v:.4f}' for k, v in self.head_loss.items()])
        return 'N/A'

    def refresh(self, bidx: int):
        '''Clear outputs to start a new batch.'''
        self.bidx = bidx                            # take input from new batch
        self.preds.clear()                          # clear the old batch
        self.total_loss = torch.empty(0)            # clear the old batch
        self.head_loss.clear()                      # clear the old batch

# ----- .epoch summary
@dataclasses.dataclass
class _EpochSummary:
    '''Epoch-level aggregates for train/val/infer.'''
    train_loss: float
    val_loss: float
    train_logs: _TrainLogs
    val_logs: _ValLogs
    infer_ctx: _InferContext

    def __str__(self) -> str:
        return '\n'.join([
            'Epoch Results:',
            f'\tTraining Loss: {self.train_loss}',
            f'\tValidation Loss: {self.val_loss}',
        ])

@dataclasses.dataclass
class _TrainLogs:
    '''Training summary for an epoch (aggregated losses).'''
    head_losses: dict[str, float]
    head_losses_str: str
    updated: bool

@dataclasses.dataclass
class _ValLogs:
    '''Validation summary for an epoch (per-head metrics).'''
    head_metrics: dict[str, dict[str, typing.Any]]
    head_metrics_str: dict[str, list[str]]

@dataclasses.dataclass
class _InferContext:
    '''Inference assembly context for block-wise stitching.'''
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]
    maps: dict[str, dict[tuple[int, int], torch.Tensor]]

# ----- .metrics
@dataclasses.dataclass
class _MetricsTracker:
    '''Experiment-level metric tracker with patience bookkeeping.'''
    last_value: float
    curr_value: float
    best_value: float
    best_epoch: int
    patience_n: int

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

# ----- .optimization state
@dataclasses.dataclass
class _OptimState:
    '''Optimization state (e.g., AMP GradScaler).'''
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

# -------------------------------Public Function-------------------------------
def init_state() -> RuntimeState:
    '''Instantiate a trainer state dataclass with placeholder values.'''

    return RuntimeState(
        progress=_Progress(
            epoch=0,
            epoch_step=0,
            global_step=0,
        ),
        heads=_Heads(
            all_heads=[],
            active_heads=None,
            frozen_heads=None,
            active_hspecs=None,
            active_hloss=None,
            active_hmetrics=None,
        ),
        batch_cxt=_BatchContex(
            bidx=0,
            pidx_start=0,
            batch_size_full=0,
            batch=None,
            x=torch.empty(0),
            y_dict={},
            domain={},
        ),
        batch_out=_BatchOutput(
            bidx=0,
            preds={},
            total_loss=torch.empty(0),
            head_loss={},
        ),
        epoch_sum=_EpochSummary(
            train_loss=0.0,
            val_loss=0.0,
            train_logs=_TrainLogs(
                head_losses={},
                head_losses_str='',
                updated=False,
            ),
            val_logs=_ValLogs(
                head_metrics={},
                head_metrics_str={},
            ),
            infer_ctx=_InferContext(
                patch_per_blk=0,
                patch_per_dim=0,
                block_columns=0,
                patch_grid_shape=(0, 0),
                maps = {}
            )
        ),
        metrics=_MetricsTracker(
            last_value=-float('inf'),
            curr_value=-float('inf'),
            best_value=-float('inf'),
            best_epoch=-1,
            patience_n=0
        ),
        optim=_OptimState()
    )
