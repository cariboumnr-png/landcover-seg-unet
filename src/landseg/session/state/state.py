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

'''Initiate a shared session-level runtime state object (mutable).'''

# standard imports
from __future__ import annotations
import dataclasses
# third-party imports
import torch
# local imports
import landseg.session.common as common
import landseg.session.common.alias as alias

# ----- Runtime state (composite)
@dataclasses.dataclass
class _RuntimeState:
    '''Composite training state with sensible defaults.'''
    progress: _Progress
    heads: _Heads
    batch_cxt: _BatchContex
    batch_out: _BatchOutput
    summary: _ResultsSummary
    optim: _OptimState

    def __str__(self):
        return '\n'.join([
            f'{str(self.progress)}',
            f'{str(self.heads)}',
            f'{str(self.batch_cxt)}',
            f'{str(self.batch_out)}',
            f'{str(self.summary)}',
            f'{str(self.optim)}'
        ])

# ----- .progress
@dataclasses.dataclass
class _Progress:
    '''Training progress counters (epoch/step/global).'''
    epoch: int
    epoch_step: int
    global_step: int
    current_metrics: float

    def __str__(self) -> str:
        return '\n'.join([
            'Progress:',
            f'\tCurrent Epoch: {self.epoch}',
            f'\tCurrent Step in Epoch: {self.epoch_step}',
            f'\tCurrent Global Step: {self.global_step}',
            f'\tCurrent Metrics: {self.current_metrics}',
        ])

# ----- .heads
@dataclasses.dataclass
class _Heads:
    '''State for multihead selection, freezing, and active specs.'''
    all_heads: list[str]
    active_heads: list[str] | None
    frozen_heads: list[str] | None
    active_hspecs: dict[str, common.SpecsLike] | None
    active_hloss: dict[str, common.CompositeLossLike] | None
    active_hmetrics: dict[str, common.ConfusionMatrixLike] | None

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

# ----- .summary
@dataclasses.dataclass
class _ResultsSummary:
    '''Summaries train/val/infer.'''
    train_summary: _TrainSummary
    val_summary: _ValSummary
    infer_context: _InferContext

    def __str__(self) -> str:
        return '\n'.join([
            'Epoch Results:',
        ])

@dataclasses.dataclass
class _TrainSummary:
    '''Training results summary.'''
    total_loss: float
    head_losses_str: str
    updated: bool

    def clear(self) -> None:
        '''Clear container.'''
        self.total_loss = 0.0
        self.head_losses_str = ''
        self.updated = False

@dataclasses.dataclass
class _ValSummary:
    '''Validation summary for an epoch (per-head metrics).'''
    target_metrics: float
    head_metrics_str: dict[str, list[str]]

    def clear(self) -> None:
        '''Clear container.'''
        self.target_metrics = 0.0
        self.head_metrics_str.clear()

@dataclasses.dataclass
class _InferContext:
    '''Inference assembly context for block-wise stitching.'''
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]
    maps: dict[str, dict[tuple[int, int], torch.Tensor]]

# ----- .optimization state
@dataclasses.dataclass
class _OptimState:
    '''Optimization state (e.g., AMP GradScaler).'''
    scaler: torch.GradScaler

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
def initialize(
    headspecs: common.HeadSpecsLike,
    dataloaders: common.DataLoadersLike,
    use_amp: bool,
    device: str,
) -> _RuntimeState:
    '''Instantiate a trainer state dataclass with placeholder values.'''

    state = _RuntimeState(
        progress=_Progress(
            epoch=0,
            epoch_step=0,
            global_step=0,
            current_metrics=0.0
        ),
        heads=_Heads(
            all_heads=list(headspecs.as_dict().keys()),
            active_heads=None,
            frozen_heads=None,
            active_hspecs=None,
            active_hloss=None,
            active_hmetrics=None,
        ),
        batch_cxt=_BatchContex(
            bidx=0,
            pidx_start=0,
            batch_size_full=dataloaders.meta.batch_size,
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
        summary=_ResultsSummary(
            train_summary=_TrainSummary(
                total_loss=0.0,
                head_losses_str='',
                updated=False,
            ),
            val_summary=_ValSummary(
                target_metrics=0.0,
                head_metrics_str={},
            ),
            infer_context=_InferContext(
                patch_per_blk=0,
                patch_per_dim=0,
                block_columns=0,
                patch_grid_shape=(0, 0),
                maps={}
            )
        ),
        optim=_OptimState(
            scaler=torch.GradScaler(
                device=device,
                enabled=use_amp
            )
        )
    )

    # if test dataset if provided, setup inference context
    if dataloaders.test:
        # resolve patch-block layout
        per_blk = dataloaders.meta.patch_per_blk
        per_dim = int(per_blk ** 0.5)
        assert per_dim * per_dim == per_blk, 'patch_per_blk must be square'
        state.summary.infer_context.patch_per_blk = per_blk
        state.summary.infer_context.patch_per_dim = per_dim
        # resolve block col/row numbers
        blk_col, blk_row = dataloaders.meta.test_blks_grid
        state.summary.infer_context.block_columns = blk_col
        # resolve patch col/row numbers
        pch_col, pch_row = (blk_col * per_dim, blk_row * per_dim)
        state.summary.infer_context.patch_grid_shape = pch_col, pch_row
    # return
    return state
