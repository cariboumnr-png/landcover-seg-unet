'''Loss calculation math blocks.'''

# third-party imports
import torch
import torch.nn
import torch.nn.functional
# local imports
import landseg.training.loss as loss

# --------------------------------Public  Class--------------------------------
class FocalLoss(loss.PrimitiveLoss):
    '''Multi-class focal loss with proper ignore_index handling.'''

    def __init__(
            self,
            alpha: list[float] | None,
            gamma: float,
            reduction: str,
            ignore_index: int
        ):
        '''
        Docstring for __init__

        :param self: Description
        :param alpha: Description
        :type alpha: list[float] | None
        :param gamma: Description
        :type gamma: float
        :param reduction: Description
        :type reduction: str
        :param ignore_index: Description
        :type ignore_index: int
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
        '''Forward.'''

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
    '''Dice loss.'''

    def __init__(self, smooth: float, ignore_index: int):
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
        '''Forward.'''

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
    Build a per-pixel weight map in [0, 1].
    - Starts at 1.0 everywhere.
    - For each (w_i, m_i) in mask, sets weight to
        min(current, clamp(w_i, 0, 1)) where m_i is True.
    - Pixels at ignore_index are forced to 0.
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
