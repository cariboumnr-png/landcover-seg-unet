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
Multi-task prediction head management for segmentation models.

This module provides centralized infrastructure for managing
multiple task-specific output heads, head selection strategies,
and per-head logit adjustment.
'''

# third-party imports
import torch
import torch.nn

class HeadManager(torch.nn.Module):
    '''
    Manage multiple 1x1 convolution heads, activation, and freezing.

    Each head is `Conv2d(in_ch, num_classes, kernel_size=1)` producing
    per-pixel logits. The manager tracks which heads are active for
    forward passes and supports freezing selected heads' parameters.
    '''

    def __init__(self, in_ch: int, heads: dict[str, int]):
        '''
        Create per-head 1x1 convs and initialize head state.

        Args:
            in_ch:
                Channel width of the shared feature map feeding all
                heads.
            heads:
                Mapping from head name to number of output classes.
        '''
        super().__init__()
        # output convolution block
        self.outc = torch.nn.ModuleDict({
            head_name: torch.nn.Conv2d(in_ch, num_classes, kernel_size=1)
            for head_name, num_classes in heads.items()
        })
        self.active: list[str] | None = list(self.outc.keys())
        self.frozen: list[str] | None = None

    def forward(
        self,
        x: torch.Tensor,
        active_heads: list[str] | None,
        logit_adjust: dict[str, torch.Tensor],
        logit_adjust_alpha: float,
    ) -> dict[str, torch.Tensor]:
        '''
        Run active heads on shared features and return per-head logits.

        Args:
            x:
                Shared backbone feature tensor of shape
                `(B, in_ch, H, W)`.
            active_heads:
                Optional list of head names to evaluate. Updates the
                stored active selection when provided.
            logit_adjust:
                Mapping from head names to prior adjustment tensors.
            logit_adjust_alpha:
                Scalar multiplier applied to logit adjustment priors.

        Returns:
            dict[str, torch.Tensor]:
                Mapping from active head names to output logit tensors.

        Raises:
            FloatingPointError:
                If NaN values are detected in any head's output logits.
        '''
        # if external active heads provided
        if active_heads is not None:
            self.active = active_heads
        # reset logic
        if self.active is None:
            self.active = list(self.outc.keys())

        # iterate through active heads
        output_logits: dict[str, torch.Tensor] = {}
        for head in self.active:
            conv = self.outc[head]
            logits = conv(x)
            if torch.isnan(logits).any():
                raise FloatingPointError(
                    f'NaN detected in output logits for head "{head}". '
                    'This indicates forward numerical corruption or an '
                    'AMP/cuDNN kernel incompatibility (see Issue #88).'
                )
            logits = torch.clamp(logits, min=-1e4, max=1e4)
            # apply logit adjustment
            output_logits[head] = self._apply_logit_adjust(
                head,
                logits,
                logit_adjust=logit_adjust,
                la_alpha=logit_adjust_alpha
            )
        return output_logits

    def freeze(self, frozen_heads: list[str] | None = None) -> None:
        '''
        Disable gradient updates for selected prediction heads.

        Args:
            frozen_heads:
                Optional list of head names to freeze. No-op when
                `None`.
        '''
        if frozen_heads is None:
            return
        for h in frozen_heads:
            for p in self.outc[h].parameters():
                p.requires_grad = False

    def unfreeze(self, unfrozen_heads: list[str] | None = None) -> None:
        '''
        Enable gradient updates for selected or all prediction heads.

        Args:
            unfrozen_heads:
                Optional list of head names to unfreeze. Unfreezes all
                registered heads when `None`.
        '''
        targets = (
            unfrozen_heads
            if unfrozen_heads is not None
            else list(self.outc.keys())
        )
        for h in targets:
            if h in self.outc:
                for p in self.outc[h].parameters():
                    p.requires_grad = True

    @staticmethod
    def _apply_logit_adjust(
        head: str,
        logits: torch.Tensor,
        *,
        logit_adjust: dict[str, torch.Tensor],
        la_alpha: float | None = None,
    ) -> torch.Tensor:
        '''Apply logit adjustment prior when enabled for this head.'''
        prior = logit_adjust.get(head)
        a = float(la_alpha) if la_alpha is not None else 1.0
        # early exit
        if prior is None or a == 0.0:
            return logits
        # apply la alpha if provided
        return logits + a * prior
