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
Loss computation blocks for segmentation models.

Currently provides implementations of:
    - FocalLoss: multi-class focal loss with ignore-index and optional
      per-pixel masking.
    - DiceLoss: soft Dice loss with weighted pixels and safe
      ignore-index handling.

Also includes a helper for constructing per-pixel weight maps from
user-provided masks and ignore-index rules.
'''

# third-party imports
import torch
import torch.nn
import torch.nn.functional
# local imports
import landseg.session.components.task.loss.primitives as primitives

# --------------------------------Public  Class--------------------------------
class DiceLoss(primitives.PrimitiveLoss):
    '''
    Multi-class soft Dice loss with ignore-index and optional masks.

    Computes per-class Dice scores from softmax probabilities and returns
    1 - mean(Dice). Supports weighted pixels through the mask mechanism.
    '''

    def __init__(
        self,
        smooth: float,
        ignore_index: int
    ):
        '''
        Initialize the Dice loss module.

        Args:
            smooth: Small constant added to the numerator and denominator
                to stabilize division, especially when masks or classes
                have zero foreground.
            ignore_index: Label to exclude completely from Dice compute.
                Pixels with this label receive zero weight and do not
                contribute to intersection or union.
        '''
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        masks: dict[float, torch.Tensor] | None,
        features: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Compute soft Dice loss across classes.

        Args:
            logits: Tensor [B, C, H, W], class logits.
            targets: Tensor [B, H, W], ground-truth labels.
            masks: Optional pixel-weight masks combined into a single
                per-pixel weight map.

        Returns:
            A scalar tensor containing the Dice loss (1 - mean Dice).
        '''

        # get per-pixel weights
        w = self._compose_pixel_weights(
            masks=masks,
            targets=targets,
            ignore_index=self.ignore_index,
            device=logits.device,
            dtype=logits.dtype
        )
        # early exit to avoid NaNs if nothing valid
        if w.sum().item() == 0:
            return logits.new_zeros(())

        # convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=1) # [N, C, H, W]

        # one-hot encode targets with ignore index handling
        targets_safe = targets.clone()
        targets_safe[targets == self.ignore_index] = 0  # any valid class
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets_safe.unsqueeze(1), 1.0)

        # apply pixel weights (broadcast over channel)
        w = w.unsqueeze(1)                                       # [N,1,H,W]
        probs_w = probs * w
        targets_w = targets_oh * w

        # standard Dice
        dims = (0, 2, 3)
        intersection = (probs_w * targets_w).sum(dims)
        union = probs_w.sum(dims) + targets_w.sum(dims)

        # stabilize
        intersection = torch.clamp(intersection, min=0.0)
        union = torch.clamp(union, min=self.smooth)

        # get dice and return
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice = torch.clamp(dice, min=0.0, max=1.0)
        return 1 - dice.mean()
