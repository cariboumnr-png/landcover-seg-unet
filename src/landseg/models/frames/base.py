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
doc
'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.backbones as backbones
import landseg.models.core as model_core

class MultiHeadBaseModel(torch.nn.Module):
    '''
    doc
    '''

    def __init__(
        self,
        *,
        body: backbones.Backbone,
        heads_manager: model_core.HeadManager,
        domain_router: model_core.DomainContextRouter,
        num_safety: model_core.NumericSafety,
      ):
        '''
        doc
        '''

        super().__init__()
        self.body = body
        self.heads = heads_manager
        self.domain_router = domain_router
        self.safety = num_safety

    @property
    def logit_adjust(self) -> dict[str, torch.Tensor]:
        '''
        Returns Lazily gather per-head logit adjustment buffers.

        Buffers are named 'la_{head}'. Excludes the scalar 'la_alpha'.
        '''
        out: dict[str, torch.Tensor] = {}
        for name, buf in self.named_buffers():
            if name.startswith('la_') and name != 'la_alpha':
                head = name.removeprefix('la_')
                out[head] = buf
        return out

    @property
    def logit_adjust_alpha(self) -> float:
        '''Returns Global logit adjust alpha scalar.'''
        return float(getattr(self, 'la_alpha').item())

    @property
    def spatial_divisor(self) -> int:
        '''Minimum spatial divisor from the model body.'''
        return self.body.spatial_divisor

    @abc.abstractmethod
    def _forward_features(
        self,
        x: torch.Tensor,
        domains: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
    ) -> torch.Tensor:
        '''Compute shared features before output heads.'''
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        ids_domain: torch.Tensor | None = None,
        vec_domain: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        '''Compute per-head logits with optional domain info.'''
        assert torch.isfinite(x).all(), 'Input has NaN/Inf'
        domain = self.domain_router.forward(ids_domain, vec_domain)
        x = self._forward_features(x, domain)
        return self.heads.forward(
            x,
            self.heads.active,
            self.logit_adjust,
            self.logit_adjust_alpha,
        )

    def set_active_heads(self, active_heads: list[str] | None=None) -> None:
        '''Set the list of active heads used during forward.'''
        self.heads.active = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None=None) -> None:
        '''Freeze parameters for selected heads.'''
        self.heads.frozen = frozen_heads
        self.heads.freeze(frozen_heads)

    def reset_heads(self):
        '''Clear active/frozen head selections.'''
        self.heads.active = None
        self.heads.frozen = None

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        '''Set logit adjust alpha.'''
        la_alpha: torch.Tensor = getattr(self, 'la_alpha')
        la_alpha.fill_(float(alpha))

    def _register_logit_adjust(self, logit_adjust: dict[str, list[float]]) -> None:
        '''doc'''
        # register logit adjustments
        # scalar strength alpha (1.0 = as-provided priors) for logit adjust
        self.register_buffer('la_alpha', torch.tensor(1.0, dtype=torch.float32))
        # register perhead logit adjustment as buffers (NOT parameters)
        for h, v in logit_adjust.items():
            t = torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer(f'la_{h}', t)
