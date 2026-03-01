# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''
Callback-facing trainer class protocols. Mimics related behaviours.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.training.common as common

# -------------------------------trainer class-------------------------------
@typing.runtime_checkable
class TrainerLike(typing.Protocol):
    comps: common.TrainerComponentsLike
    config: common.RuntimeConfigLike
    state: common.RuntimeStateLike
    flags: dict[str, bool]
    device: str
    # batch extraction
    def _parse_batch(self) -> None: ...
    # context
    def _autocast_ctx(self) -> typing.ContextManager: ...
    def _val_ctx(self) -> typing.ContextManager: ...
    # training pahse
    def _compute_loss(self) -> None: ...
    def _clip_grad(self) -> None: ...
    def _update_train_logs(self, flush: bool=True) -> bool: ...
    # validation phase
    def _update_conf_matrix(self) -> None: ...
    def _compute_iou(self) -> None: ...
    def _track_metrics(self) -> None: ...
    # inference phase
    def _aggregate_batch_predictions(self) -> None: ...
    def _preview_monitor_head(self, out_dir: str) -> None: ...
