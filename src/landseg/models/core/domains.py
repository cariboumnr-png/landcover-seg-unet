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

'''
Domain conditioning routing utilities.

This module provides lightweight infrastructure for routing raw domain
conditioning inputs to named architecture targets.

Supported conditioning sources include:
- discrete domain identifiers
- continuous domain vectors

Each target independently defines:
- whether identifier embeddings are used
- whether vector projections are used
- the projection architecture for vector conditioning

The router itself is architecture-agnostic and only produces routed
conditioning payloads. Consumption of routed payloads is delegated to
downstream conditioning modules (e.g., concatenation adapters, FiLM
layers, normalization conditioning, attention modulation).
'''

# standard imports
import dataclasses
# third-party imports
import torch
# local imports
import landseg.models.core.config as config

@dataclasses.dataclass(frozen=True)
class DomainTargetPayload:
    '''Container for routed domain tensors.'''
    ids_embd: torch.Tensor | None
    vec_proj: torch.Tensor | None

class DomainContextRouter(torch.nn.Module):
    '''
    Route raw domain conditioning inputs to named targets.

    Each target may independently:
    - embed discrete domain identifiers
    - project continuous domain vectors
    - configure its own projection architecture

    Target outputs are returned as structured payloads keyed by target
    name. This module does not apply conditioning directly and remains
    agnostic to downstream architecture implementations.
    '''

    def __init__(
        self,
        *,
        domain_ids_num: int,
        domain_vec_dim: int,
        targets: dict[str, config.DomainTargetConfig],
    ) -> None:
        '''
        Initialize target-specific domain conditioning modules.

        Embedding tables and vector projection modules are constructed
        only for targets that explicitly enable the corresponding
        conditioning source.

        Args:
            domain_ids_num: Total number of discrete domain identifiers.
                Required when one or more targets enable identifier
                embeddings.

            domain_vec_dim: Input dimensionality of continuous domain
                vectors. Required when one or more targets enable vector
                projections.

            targets: Mapping of target names to target-specific
                conditioning configurations.
        '''

        super().__init__()
        self.targets = targets
        self.ids_embd = torch.nn.ModuleDict()
        self.vec_proj = torch.nn.ModuleDict()

        for name, target_cfg in targets.items():

            if target_cfg.use_vec and domain_vec_dim > 0:
                self.vec_proj[name] = _make_projection(
                    domain_vec_dim,
                    target_cfg.vec_proj_dims,
                    projection_config=target_cfg.vec_proj_config
                )

            if target_cfg.use_ids and domain_ids_num > 0:
                self.ids_embd[name] = torch.nn.Embedding(
                    domain_ids_num,
                    target_cfg.ids_embd_dims
                )

    def forward(
        self,
        ids: torch.Tensor | None,
        vec: torch.Tensor | None,
    ) -> dict[str, DomainTargetPayload]:
        '''
        Route domain conditioning inputs to configured targets.

        Args:
            ids: Tensor containing discrete domain identifiers. May be
                omitted if identifier conditioning is unused.

            vec: Tensor containing continuous domain vectors. May be
                omitted if vector conditioning is unused.

        Returns:
            dict:
                Mapping of target names to routed conditioning payloads.

            Each payload may contain:
            - embedded domain identifiers
            - projected domain vectors

            Individual payload fields are `None` when the corresponding
            conditioning source is unavailable or disabled for the target.
        '''

        routed: dict[str, DomainTargetPayload] = {}
        for name, _ in self.targets.items():

            if ids is not None and name in self.ids_embd:
                target_ids_embd = self.ids_embd[name](ids)
            else:
                target_ids_embd = None

            if vec is not None and name in self.vec_proj:
                target_vec_proj = self.vec_proj[name](vec)
            else:
                target_vec_proj = None

            routed[name] = DomainTargetPayload(target_ids_embd, target_vec_proj)
        return routed

# ---------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------
def _make_projection(
    in_dim: int,
    out_dim: int,
    *,
    projection_config: config.DomainProjectionConfig,
) -> torch.nn.Module:
    '''Build a linear or MLP projection for domain vectors.'''

    # parse configs with defaults
    use_mlp = projection_config.get('use_mlp', False)
    hidden_dim = projection_config.get('hidden_dim', 0) or out_dim
    num_layers = projection_config.get('num_hidden_layers', 2)
    dropout = projection_config.get('dropout', 0.0)
    activation = projection_config.get('activation', 'silu')

    # return a simple linear mapping (e.g., concat without mlp)
    if not use_mlp:
        return torch.nn.Linear(in_dim, out_dim)

    # width of the hidden layers
    layers: list[torch.nn.Module] = []
    curr_dim = in_dim
    # build hidden layers
    for _ in range(max(num_layers, 1)):
        layers.append(torch.nn.Linear(curr_dim, hidden_dim))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        curr_dim = hidden_dim

    # final projection to the output dimension
    layers.append(torch.nn.Linear(curr_dim, out_dim))
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
