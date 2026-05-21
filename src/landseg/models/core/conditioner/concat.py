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
Spatial domain-feature concatenation adapters.

This module provides lightweight conditioning adapters that consume
routed domain conditioning payloads and concatenate them to convolutional
feature maps as additional channels.

Conditioning tensors are produced externally by a
`DomainContextRouter`. This module is intentionally limited to:
    - consuming routed payloads
    - broadcasting conditioning tensors spatially
    - concatenating conditioning channels to feature maps

Projection, embedding, and routing logic are delegated to the routing
subsystem.
'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.core as model_core

class ConcatAdapter(torch.nn.Module):
    '''
    Concatenate routed domain features to spatial feature maps.

    This adapter consumes precomputed domain conditioning produced by
    `DomainContextRouter`. It does not perform embedding or projection.

    Supported inputs:
        - ids_embd: embedded domain identifiers [B, D]
        - vec_proj: projected domain vectors [B, D]

    If both are provided, they are summed.

    Args:
        out_dom: Number of domain channels expected for concatenation.

    Raises:
        ValueError: If payload dimensions do not match `out_dom`.
    '''

    def __init__(
        self,
        *,
        concat_dim: int,
    ) -> None:
        '''
        Initialize the adapter.

        Args:
            out_dom: Number of domain channels to append.
        '''
        super().__init__()
        self.output_dim = concat_dim

    def forward(
        self,
        x: torch.Tensor,
        payload: model_core.DomainTargetPayload | None,
    ) -> torch.Tensor:
        '''
        Concatenate domain conditioning to input tensor.

        Args:
            x: Input tensor [B, C, H, W].
            payload: Routed domain payload for this target.

        Returns:
            Tensor: [B, C + out_dom, H, W] if out_dom > 0.

        Raises:
            ValueError: If payload is missing or invalid when required.
        '''
        if self.output_dim <= 0:
            return x

        if payload is None:
            raise ValueError('Expected DomainTargetPayload, got None')

        dv_ids = payload.ids_embd
        dv_vec = payload.vec_proj

        if dv_ids is None and dv_vec is None:
            raise ValueError(
                'Payload contains neither ids_embd nor vec_proj'
            )

        dv: torch.Tensor | None = None

        if dv_ids is not None and dv_vec is not None:
            if dv_ids.shape[1] != dv_vec.shape[1]:
                raise ValueError('Mismatched payload dimensions')
            dv = dv_ids + dv_vec
        else:
            dv = dv_ids if dv_ids is not None else dv_vec

        assert dv is not None

        if dv.shape[1] != self.output_dim:
            raise ValueError(
                f'Expected domain dim={self.output_dim}, '
                f'got {dv.shape[1]}'
            )

        b, _, h, w = x.shape
        dv = dv.to(x.device)
        dv = dv.view(b, self.output_dim, 1, 1)
        dv = dv.expand(b, self.output_dim, h, w)

        return torch.cat([x, dv], dim=1)
