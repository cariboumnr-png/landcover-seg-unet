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
'''Multihead model protocol.'''

# standard imports
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import torch
    import torch.nn

class MultiheadModelLike(typing.Protocol):
    '''Minimally required class methods.'''
    def __call__(self, x: 'torch.Tensor', **kwargs) -> typing.Mapping[str, 'torch.Tensor']: ...
    def forward(self, x: 'torch.Tensor', **kwargs) -> typing.Mapping[str, 'torch.Tensor']: ...
    def parameters(self) -> typing.Iterable['torch.nn.Parameter']: ...
    def to(self: typing.Self, device: 'torch.device | str') -> typing.Self: ...
    def train(self: typing.Self, mode: bool = True) -> typing.Self: ...
    def eval(self: typing.Self) -> typing.Self: ...
    def set_active_heads(self, active_heads: list[str] | None) -> None: ...
    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None: ...
    def reset_heads(self) -> None: ...
    def set_logit_adjust_enabled(self, enabled: bool) -> None: ...
    def set_logit_adjust_alpha(self, alpha: float) -> None: ...
    def state_dict(self) -> typing.Mapping[str, 'torch.Tensor']: ...
    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) -> typing.Any: ...
