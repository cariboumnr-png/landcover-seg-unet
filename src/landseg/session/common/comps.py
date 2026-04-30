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
Session components protocols
'''

# standard imports
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import torch

# -----------------------------Engine components-----------------------------
@typing.runtime_checkable
class ComponentsLike(typing.Protocol):
    @property
    def dataloaders(self) -> DataLoadersLike:...
    @property
    def headspecs(self) -> HeadSpecsLike:...
    @property
    def headlosses(self) -> _HeadLossesLike:...
    @property
    def headmetrics(self) -> _HeadMetricsLike:...
    @property
    def optimization(self) -> _OptimizationLike:...

# ---------------------------------dataloaders---------------------------------
@typing.runtime_checkable
class DataLoadersLike(typing.Protocol):
    @property
    def train(self) -> 'torch.utils.data.DataLoader | None':...
    @property
    def val(self) -> 'torch.utils.data.DataLoader | None':...
    @property
    def test(self) -> 'torch.utils.data.DataLoader | None':...
    @property
    def preview_context(self) -> '_PreviewContext | None': ...

class _PreviewContext(typing.Protocol):
    patch_per_blk: int
    patch_per_dim: int
    block_columns: int
    patch_grid_shape: tuple[int, int]

# ---------------------------------head  specs---------------------------------
@typing.runtime_checkable
class HeadSpecsLike(typing.Protocol):
    def __getitem__(self, key: str) -> SpecsLike: ...
    def __len__(self) -> int: ...
    def as_dict(self) -> typing.Mapping[str, SpecsLike]: ...

@typing.runtime_checkable
class SpecsLike(typing.Protocol):
    @property
    def name(self) -> str:...
    @property
    def count(self) -> list[int]:...
    @property
    def loss_alpha(self) -> list[float]:...
    @property
    def parent_head(self) -> str | None:...
    @property
    def parent_cls(self) -> int | None:...
    @property
    def weight(self) -> float:...
    @property
    def exclude_cls(self) -> tuple[int, ...] | None:...

# ----------------------------------head loss----------------------------------
@typing.runtime_checkable
class _HeadLossesLike(typing.Protocol):
    def __getitem__(self, key: str) -> CompositeLossLike: ...
    def __len__(self) -> int: ...
    def as_dict(self) -> typing.Mapping[str, CompositeLossLike]: ...

@typing.runtime_checkable
class CompositeLossLike(typing.Protocol):
    @property
    def ignore_index(self) -> int:...
    def forward(self, p: 'torch.Tensor', t: 'torch.Tensor', **kwargs) -> 'torch.Tensor': ...

# --------------------------------head  metrics--------------------------------
@typing.runtime_checkable
class _HeadMetricsLike(typing.Protocol):
    def __getitem__(self, key: str) -> ConfusionMatrixLike: ...
    def __len__(self) -> int: ...
    def as_dict(self) -> typing.Mapping[str, ConfusionMatrixLike]: ...

@typing.runtime_checkable
class ConfusionMatrixLike(typing.Protocol):
    @property
    def metrics(self) -> AccumulatedMetrics: ...
    def update(self, preds: 'torch.Tensor', targets: 'torch.Tensor', **kwargs) -> None: ...
    def compute(self) -> None: ...
    def reset(self, device: str) -> None: ...

@typing.runtime_checkable
class AccumulatedMetrics(typing.Protocol):
    @property
    def mean(self) -> float: ...
    @property
    def ious(self) -> dict[str, float]: ...
    @property
    def support(self) -> dict[str, int]: ...
    @property
    def ac_mean(self) -> float: ...
    @property
    def ac_ious(self) -> dict[str, float]: ...
    @property
    def ac_support(self) -> dict[str, int]: ...
    @property
    def as_dict(self) -> dict[str, typing.Any]: ...
    @property
    def as_str_list(self) -> list[str]: ...

# ------------------------------------optim------------------------------------
@typing.runtime_checkable
class _OptimizationLike(typing.Protocol):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
