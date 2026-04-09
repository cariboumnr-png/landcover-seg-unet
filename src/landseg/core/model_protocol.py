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

# pylint: disable=missing-function-docstring

'''
Multihead model protocol.

This module defines the structural interface for multihead models used
throughout the project.

The `MultiheadModelLike` protocol specifies the expected behavior and API
surface of models that expose multiple prediction heads (e.g., for
multi-task or hierarchical learning).

Models conforming to this protocol are:
    - **Constructed** in `models` module, where concrete architectures
      implement this interface.
    - **Passed across the CLI boundary** at runtime into trainers and
      evaluators.
    - **Consumed** by training and evaluation pipelines, which rely on
      this protocol for consistent interaction (forward pass, head
      control, state management, etc.).

As part of the `./core/` package, this protocol acts as a contract
between model implementations and downstream runtime components,
enabling decoupled development and interchangeability of model backends.
'''

# standard imports
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import torch
    import torch.nn

    # aliases
    Tensor: typing.TypeAlias = torch.Tensor

# ---------------------------------Public Type---------------------------------
class MultiheadModelLike(typing.Protocol):
    '''
    Protocol defining the required interface for multihead models.

    A conforming model exposes multiple output heads, each producing a
    tensor keyed by head name. The protocol standardizes how models are
    invoked, configured, and managed during training and evaluation.

    Key capabilities include:
        - Forward execution returning a mapping of head names to outputs
        - Device placement and training/evaluation mode control
        - Selective activation and freezing of heads
        - Optional logit adjustment configuration for imbalance handling
        - Compatibility with PyTorch state management (`state_dict`)

    This interface enables trainers and evaluators to operate on any
    compliant model without depending on specific implementations,
    ensuring flexibility and modularity across the system.
    '''
    def __call__(self, x: 'Tensor', **kwargs) -> typing.Mapping[str, 'Tensor']: ...
    def forward(self, x: 'Tensor', **kwargs) -> typing.Mapping[str, 'Tensor']: ...
    def parameters(self) -> typing.Iterable['torch.nn.Parameter']: ...
    def to(self: typing.Self, device: 'torch.device | str') -> typing.Self: ...
    def train(self: typing.Self, mode: bool = True) -> typing.Self: ...
    def eval(self: typing.Self) -> typing.Self: ...
    def set_active_heads(self, active_heads: list[str] | None) -> None: ...
    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None: ...
    def reset_heads(self) -> None: ...
    def set_logit_adjust_enabled(self, enabled: bool) -> None: ...
    def set_logit_adjust_alpha(self, alpha: float) -> None: ...
    def state_dict(self) -> typing.Mapping[str, 'Tensor']: ...
    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) -> typing.Any: ...
