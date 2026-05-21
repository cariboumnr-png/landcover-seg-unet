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
FiLM-based domain conditioning modules.

This module provides FiLM conditioning adapters that consume routed
domain conditioning payloads and generate feature-wise affine modulation
parameters for convolutional bottleneck features.

Conditioning tensors are produced externally by a
`DomainContextRouter`. This module is intentionally limited to:
    - consuming routed conditioning payloads
    - generating FiLM modulation parameters
    - applying feature-wise affine conditioning

Embedding, projection, and routing logic are delegated to the routing
subsystem.
'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.core as model_core


class FilmConditioner(torch.nn.Module):
    '''
    FiLM-based modulation using routed domain conditioning.

    This module consumes precomputed domain representations from
    `DomainContextRouter` and produces FiLM parameters to modulate
    feature maps.

    Supported inputs:
        - ids_embd: [B, D]
        - vec_proj: [B, D]

    If both are present, they are summed before FiLM generation.
    '''

    def __init__(
        self,
        *,
        embed_dim: int,
        bottleneck_ch: int,
        hidden_dim: int | None = None,
    ) -> None:
        '''
        Initialize FiLM generator.

        Args:
            embed_dim: Expected domain embedding dimension (D).
            bottleneck_ch: Number of feature channels to modulate.
            hidden: Hidden width of FiLM MLP (defaults to embed_dim).
        '''
        super().__init__()

        self.embed_dim = embed_dim

        if embed_dim <= 0:
            self.film = None
            return

        h = hidden_dim or embed_dim

        self.film = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, h),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(h, 2 * bottleneck_ch),
        )

        for module in self.film.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def _build_z(
        self,
        payload: model_core.DomainTargetPayload | None,
    ) -> torch.Tensor | None:
        '''
        Construct combined domain embedding z.

        Args:
            payload: Routed domain payload.

        Returns:
            Tensor [B, D] or None.
        '''
        if payload is None or self.embed_dim <= 0:
            return None

        z_ids = payload.ids_embd
        z_vec = payload.vec_proj

        if z_ids is None and z_vec is None:
            return None

        if z_ids is not None and z_vec is not None:
            if z_ids.shape[1] != z_vec.shape[1]:
                raise ValueError('Mismatched payload dimensions')
            z = z_ids + z_vec
        else:
            z = z_ids if z_ids is not None else z_vec

        assert z is not None

        if z.shape[1] != self.embed_dim:
            raise ValueError(
                f'Expected embedding dim={self.embed_dim}, '
                f'got {z.shape[1]}'
            )

        return torch.nn.functional.layer_norm(z, (z.shape[-1],))

    def forward(
        self,
        xb: torch.Tensor,
        payload: model_core.DomainTargetPayload | None,
    ) -> torch.Tensor:
        '''
        Apply FiLM modulation using routed domain payload.

        Args:
            xb: Feature map [B, C, H, W]
            payload: Routed domain conditioning

        Returns:
            Modulated tensor (same shape as xb)
        '''
        if self.embed_dim <= 0 or self.film is None:
            return xb

        z = self._build_z(payload)
        if z is None:
            return xb

        gb = self.film(z)                 # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)  # [B, C]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return xb * (1.0 + gamma) + beta
