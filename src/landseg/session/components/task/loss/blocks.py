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
import landseg.session.components.task.loss as loss

# --------------------------------Public  Class--------------------------------
class FocalLoss(loss.PrimitiveLoss):
    '''
    Multi-class focal loss supporting per-pixel weights and ignore_index.

    The loss operates on per-pixel logits, optionally reweights pixels
    through a mask dictionary, and applies the standard focal formulation:

        FL = - a_t * (1 - p_t) ^ y * log(p_t)

    where a_t is class weighting (optional) and y controls focus on hard
    examples.
    '''

    def __init__(
        self,
        alpha: list[float] | None,
        gamma: float,
        reduction: str,
        ignore_index: int
    ):
        '''
        Initialize a multi-class focal loss module.

        Args:
            alpha: Optional list of per-class weights. If None, all classes
                receive equal weight.
            gamma: Focal exponent controlling down-weighting of easy
                samples.
            reduction: One of {'mean', 'sum', 'none'}, applied after
                masking.
            ignore_index: Label to exclude entirely from loss compute.
        '''
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        masks: dict[float, torch.Tensor] | None
    ) -> torch.Tensor:
        '''
        Compute focal loss over valid pixels.

        Args:
            logits: Tensor of shape [B, C, H, W], raw class logits.
            targets: Tensor of shape [B, H, W], integer class labels.
            masks: Optional dict mapping weights (floats in [0,1]) to
                bool/float masks selecting which pixels to down-weight.

        Returns:
            A scalar loss tensor (after reduction).
        '''

        # logits: [B, C, H, W]
        # targets: [B, H, W]
        _, c, _, _ = logits.shape # get number of classes (c)

        # get per-pixel weights
        w = _compose_pixel_weights(
            masks=masks,
            targets=targets,
            ignore_index=self.ignore_index,
            device=logits.device,
            dtype=logits.dtype
        )
        # validity: only compute for pixels with weight > 0 (e.g., valid labels)
        valid = w > 0

        # early exit if all pixel weights are 0
        if valid.sum().item() == 0:
            return logits.new_zeros(())

        # flatten N = B × H × W
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c)     # [N, C]
        targets_flat = targets.reshape(-1)                          # [N]
        w_flat = w.reshape(-1)                                      # [N]
        # select valid entries: M -> number of valid pixels
        logits_flat = logits_flat[valid.reshape(-1)]                # [M, C]
        targets_flat = targets_flat[valid.reshape(-1)]              # [M]
        w_flat = w_flat[valid.reshape(-1)]                          # [M]

        # sanity - no index out of bounds
        assert targets_flat.max() < c, \
            f'Invalid target index: {targets_flat.max()} >= {c}'

        # logits -> probabilities with clamp to avoid extreme grads
        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=1)
        log_probs = torch.clamp(log_probs, min=-30.0, max=30.0)
        probs = log_probs.exp()

        # select true class probabilities
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)# [M]
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)        # [M]

        # clamp again for stability
        log_pt = torch.clamp(log_pt, min=-20.0)  # avoid -inf
        pt = torch.clamp(pt, min=1e-6)           # avoid 0

        # alpha weighting
        if self.alpha is None:
            alpha_t = 1.0 # no effect
        else:
            alpha_t = torch.tensor(self.alpha).to(logits.device)[targets_flat]

        # focal loss weighted by pixel weights
        weighted = -alpha_t * (1 - pt).pow(self.gamma) * log_pt * w_flat # [M]
        if self.reduction == 'mean':
            return weighted.sum() / w_flat.sum()
        if self.reduction == 'sum':
            return weighted.sum()
        return weighted

class DiceLoss(loss.PrimitiveLoss):
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
        masks: dict[float, torch.Tensor] | None
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
        w = _compose_pixel_weights(
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

# ------------------------------private  function------------------------------
def _compose_pixel_weights(
    *,
    masks: dict[float, torch.Tensor] | None,
    targets: torch.Tensor,
    ignore_index: int | None,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    '''
    Construct a per-pixel weight map in [0, 1].

    Behavior:
        - Start with weight 1.0 for all pixels.
        - For each (w_i, mask_i), update weights:
              weight[pixel] = min(weight, clamp(w_i, 0, 1))
          where mask_i is True.
        - Pixels at ignore_index receive weight 0.
    '''

    # init weight tensor aligned with the targets
    ws = torch.ones_like(targets, dtype=dtype, device=device)

    # if mask dict is provided
    if masks is not None:
        for w, m in masks.items():
            # sanity checks
            assert isinstance(w, (float, int)), f'Invalid mask keys {w}'
            assert isinstance(m, torch.Tensor), f'Invalid mask type: {m}'
            assert m.shape == targets.shape, f'{m.shape} != {targets.shape}.'
            # ensure mask as bool (could be float initially)
            m_bool = m.to(dtype=torch.bool)
            # clamp to [0, 1] for down-weighting semantics
            w_val = float(max(0.0, min(1.0, w)))
            # assign weights as minimum per pixel
            if w_val < 1.0:
                ws = torch.where(
                    condition=m_bool,
                    input=torch.minimum(ws, torch.full_like(ws, w_val)),
                    other=ws
                )
            # if w_val == 1.0, no change; (>1.0 is clamped)

    # hard zeroing on ignore index
    if ignore_index is not None:
        ws = torch.where(targets == ignore_index, torch.zeros_like(ws), ws)

    # return weight map
    return ws
