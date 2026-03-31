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
Callback-facing trainer class protocols. Mimics related behaviours.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.core as core
import landseg.trainer.common as common

# -------------------------------trainer class-------------------------------
@typing.runtime_checkable
class TrainerEngineLike(typing.Protocol):
    model: core.MultiheadModelLike
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
