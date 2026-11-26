# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

Provides lightweight conditioning adapters that consume routed domain
conditioning payloads and concatenate them to convolutional feature maps
as additional channels.

Conditioning tensors are produced externally by a
`DomainContextRouter`. This module is intentionally limited to
consuming routed payloads, broadcasting conditioning tensors spatially,
and concatenating conditioning channels to feature maps.
'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.core.domains as domains

class ConcatAdapter(torch.nn.Module):
    '''
    Concatenate routed domain features to spatial feature maps.

    Consumes precomputed domain conditioning produced by
    `DomainContextRouter` without performing embedding or projection.
    When both `ids_embd` and `vec_proj` are provided, they are summed.
    '''

    def __init__(self, *, concat_dim: int) -> None:
        '''
        Initialize the concatenation adapter with output dimension.

        Args:
            concat_dim:
                Number of domain channels to concatenate to feature
                maps.
        '''
        super().__init__()
        self.output_dim = concat_dim

    def forward(
        self,
        x: torch.Tensor,
        payload: domains.DomainTargetPayload | None,
    ) -> torch.Tensor:
        '''
        Concatenate domain conditioning channels to input tensor.

        Args:
            x:
                Input feature tensor of shape `(B, C, H, W)`.
            payload:
                Routed domain conditioning payload for this target, or
                `None` when `concat_dim <= 0`.

        Returns:
            torch.Tensor:
                Tensor of shape `(B, C + concat_dim, H, W)` when
                `concat_dim > 0`, otherwise `x` unchanged.

        Raises:
            ValueError:
                If `concat_dim > 0` and `payload` is `None`, empty,
                has mismatched embedding dimensions, or does not match
                `concat_dim`.
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
