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
    Manage multiple 1x1 conv heads, activation state, and
    freezing.

    Each head is Conv2d(in_ch → num_classes, kernel_size=1)
    producing per-pixel logits. The manager tracks which heads
    are active for forward passes and supports freezing
    selected heads' parameters.
    '''

    def __init__(self, in_ch: int, heads: dict[str, int]):
        '''
        Create per-head 1x1 convs and initialize head state.

        Args:
            in_ch: Channel of the shared feature map feeding all heads.
            heads: Mapping from head's name to head's number of classes.
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
        '''Run active heads and return a dict of logits.'''

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
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            # apply logit adjustment
            output_logits[head] = self._apply_logit_adjust(
                head,
                logits,
                logit_adjust=logit_adjust,
                la_alpha=logit_adjust_alpha
            )
        return output_logits

    def freeze(self, frozen_heads: list[str] | None = None) -> None:
        '''Disable gradients of selected heads.'''
        if frozen_heads is None:
            return
        for h in frozen_heads:
            for p in self.outc[h].parameters():
                p.requires_grad = False

    @staticmethod
    def _apply_logit_adjust(
        head: str,
        logits: torch.Tensor,
        *,
        logit_adjust: dict[str, torch.Tensor],
        la_alpha: float | None = None,
    ) -> torch.Tensor:
        '''
        Apply logit adjustment if enabled and available for this head.
        '''

        prior = logit_adjust.get(head)
        a = float(la_alpha) if la_alpha is not None else 1.0
        # early exit
        if prior is None or a == 0.0:
            return logits
        # apply la alpha if provided
        return logits + a * prior
