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

'''doc'''

# standard imports
from __future__ import annotations
import typing
# third-party imports
import torch

class DomainTargetConfig(typing.Protocol):
    '''Typed container for configuring concatenation adapter.'''
    @property
    def use_ids(self) -> bool: ...
    @property
    def use_vec(self) -> bool: ...
    @property
    def projection(self) -> DomainProjectionConfig: ...

class DomainProjectionConfig(typing.Protocol):
    '''Projection configuration for one domain target.'''
    @property
    def out_dim(self) -> int: ...
    @property
    def use_mlp(self) -> bool: ...
    @property
    def hidden_dim(self) -> int | None: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def dropout(self) -> float: ...
    @property
    def activation(self) -> str: ...

class DomainContextRouter(torch.nn.Module):
    '''
    Route raw domain information to named conditioning targets.

    This module is architecture-agnostic. Each target may optionally
    define its own vector projection.
    '''

    def __init__(
        self,
        *,
        domain_vec_dim: int | None,
        targets: dict[str, DomainTargetConfig],
    ) -> None:
        '''Initialize target-specific vector projections.'''

        super().__init__()
        self.targets = targets
        self.proj = torch.nn.ModuleDict()

        if domain_vec_dim is None:
            return
        for name, cfg in targets.items():
            if not cfg.use_vec or cfg.projection is None:
                continue
            self.proj[name] = _make_projection(domain_vec_dim, cfg.projection)

    def forward(
        self,
        ids: torch.Tensor | None,
        vec: torch.Tensor | None,
    ) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
        '''Route domain inputs to configured target payloads.'''

        routed: dict[str, tuple] = {}
        for name, cfg in self.targets.items():
            target_ids = ids if cfg.use_ids else None
            target_vec = None

            if cfg.use_vec and vec is not None:
                if name in self.proj:
                    target_vec = self.proj[name](vec)
                else:
                    target_vec = vec
            # (ids, vec)
            routed[name] = (target_ids, target_vec)
        return routed

#
def _make_projection(
    in_dim: int,
    cfg: DomainProjectionConfig,
) -> torch.nn.Module:
    '''Build a linear or MLP projection for domain vectors.'''

    if not cfg.use_mlp:
        return torch.nn.Linear(in_dim, cfg.out_dim)

    hidden_dim = cfg.hidden_dim or max(in_dim, cfg.out_dim)

    layers: list[torch.nn.Module] = []
    curr_dim = in_dim

    for _ in range(max(cfg.num_layers - 1, 1)):
        layers.append(torch.nn.Linear(curr_dim, hidden_dim))
        layers.append(_get_activation(cfg.activation))

        if cfg.dropout > 0.0:
            layers.append(torch.nn.Dropout(cfg.dropout))

        curr_dim = hidden_dim

    layers.append(torch.nn.Linear(curr_dim, cfg.out_dim))
    return torch.nn.Sequential(*layers)

def _get_activation(name: str) -> torch.nn.Module:
    '''Return an activation module by name.'''

    if name == 'relu':
        return torch.nn.ReLU()
    if name == 'gelu':
        return torch.nn.GELU()
    if name == 'silu':
        return torch.nn.SiLU()
    raise ValueError(f'Unsupported activation: {name}')

# example config
_ = '''
    conditioning:
        targets:
            input:
                use_ids: true
                use_vec: true
                projection:
                    out_dim: 8
                    use_mlp: false

            bottleneck:
                use_ids: true
                use_vec: true
                projection:
                    out_dim: 64
                    use_mlp: true
                    hidden_dim: 128
                    num_layers: 2
                    activation: gelu
                    dropout: 0.1

            tokens:
                use_ids: true
                use_vec: true
                projection:
                    out_dim: 256
                    use_mlp: true
                    hidden_dim: 512
                    num_layers: 2
                    activation: gelu
    '''
