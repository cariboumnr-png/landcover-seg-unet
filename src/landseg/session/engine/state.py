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
import landseg.session.common.alias as alias
import landseg.session.engine.protocols as protocols

# alias
field = dataclasses.field

# ----- progress tracking
@dataclasses.dataclass
class _Progress:
    '''Training progress counters (epoch/step/global).'''
    epoch: int = 0
    global_step: int = 0

# ----- heads management
@dataclasses.dataclass
class _Heads:
    '''State for multihead selection, freezing, and active specs.'''
    all_heads: list[str] = field(default_factory=list)
    active_heads: list[str] | None = None
    frozen_heads: list[str] | None = None
    active_hspecs: dict[str, protocols.SpecsLike] | None = None
    active_hloss: dict[str, protocols.CompositeLossLike] | None = None
    active_hmetrics: dict[str, protocols.ConfusionMatrixLike] | None = None

# ----- batch context
@dataclasses.dataclass
class _BatchContex:
    '''Per-batch input/context (indices, tensors, and domain info).'''
    bidx: int = 0
    pidx_start: int = 0
    batch_size: int = 0 #
    batch: alias.DatasetItem | None = None
    x: torch.Tensor = torch.empty(0)
    y_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    domain: dict[str, torch.Tensor | None] = field(default_factory=dict)

    def refresh(self, bidx: int, batch: tuple) -> None:
        '''Reset batch context for a new iteration.'''
        # take input from new batch
        self.bidx = bidx
        self.batch = batch
        # calc starting patch id of this batch
        self.pidx_start = (bidx - 1) * self.batch_size
        # clear old batch
        self.x = torch.empty(0)
        self.y_dict.clear()
        self.domain.clear()

# ----- batch output
@dataclasses.dataclass
class _BatchOutput:
    '''Per-batch outputs: predictions and losses.'''
    bidx: int = 0
    preds: dict[str, torch.Tensor] = field(default_factory=dict)
    total_loss: torch.Tensor = torch.empty(0)
    head_loss: dict[str, float] = field(default_factory=dict)
    infer_maps: dict[str, dict[tuple[int, int], torch.Tensor]] = field(default_factory=dict)

    def refresh(self, bidx: int):
        '''Clear outputs to start a new batch.'''
        self.bidx = bidx                            # take input from new batch
        self.preds.clear()                          # clear the old batch
        self.total_loss = torch.empty(0)            # clear the old batch
        self.head_loss.clear()                      # clear the old batch
        # note: we do not clear inference results mapping (batch aggregation)

# ----- optimization runtime status
@dataclasses.dataclass
class _OptimRuntime:
    '''Runtime optimizer state (AMP scaler and LR snapshot).'''
    scaler: torch.GradScaler
    lrs: list[float] = field(default_factory=list)

    @property
    def lr(self) -> float | None:
        '''Return the primary Learning Rate value.'''
        return self.lrs[0] if self.lrs else None

# ----- Runtime state (composite)
@dataclasses.dataclass
class EngineState:
    '''Composite training state with sensible defaults.'''
    optim: _OptimRuntime
    progress: _Progress = field(default_factory=_Progress)
    heads: _Heads = field(default_factory=_Heads)
    batch_cxt: _BatchContex = field(default_factory=_BatchContex)
    batch_out: _BatchOutput = field(default_factory=_BatchOutput)

# -------------------------------Public Function-------------------------------
def initialize_state(
    *,
    all_heads: list[str],
    batch_size: int,
    use_amp: bool,
    device: str,
) -> EngineState:
    '''Instantiate a trainer state dataclass with placeholder values.'''

    # create an instance with default values
    runtime_state = EngineState(
        optim=_OptimRuntime(
            scaler=torch.GradScaler(device=device, enabled=use_amp)
        )
    )
    # update
    runtime_state.heads.all_heads = all_heads
    runtime_state.batch_cxt.batch_size = batch_size

    # return
    return runtime_state
