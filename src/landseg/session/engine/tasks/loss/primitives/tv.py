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
Spectral smoothness loss.

This loss encourages neighboring pixels with similar features to have
similar predicted class probabilities.
'''

# third-party imports
import torch
import torch.nn.functional
# local imports
import landseg.session.components.task.loss.primitives as primitives

class TotalVariationLoss(primitives.PrimitiveLoss):
    '''
    Total Variation (TV) loss for spatial smoothness.

    Encourages neighboring pixels to have similar class probabilities.
    Operates on softmax-normalized logits.

    Notes:
    - Independent of input features (not spectral-aware).
    - Uses targets only for optional masking / ignore handling.
    '''

    def __init__(self, ignore_index: int | None = None):
        super().__init__()
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
        Compute TV loss over spatial dimensions.

        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
            masks: optional weighting masks
            features: unused

        Returns:
            Scalar TV loss
        '''

        assert logits.dim() == 4, f'Expected (B, C, H, W), got {logits.shape}'

        # convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)

        # compute spatial differences
        dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])  # vertical
        dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])  # horizontal

        # build pixel weights (B, H, W)
        ws = self._compose_pixel_weights(
            masks=masks,
            targets=targets,
            ignore_index=self.ignore_index,
            device=logits.device,
            dtype=logits.dtype
        )

        # match shapes for dh and dw
        ws_h = ws[:, 1:, :]   # aligns with dh
        ws_w = ws[:, :, 1:]   # aligns with dw

        # expand weights to channel dimension
        ws_h = ws_h.unsqueeze(1)  # (B, 1, H-1, W)
        ws_w = ws_w.unsqueeze(1)  # (B, 1, H, W-1)

        # apply weights
        dh = dh * ws_h
        dw = dw * ws_w

        # normalize by number of valid pixels
        denom = ws_h.sum() + ws_w.sum()
        if denom > 0:
            loss = (dh.sum() + dw.sum()) / denom
        else:
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        return loss

# overfit test to show implementation success
# same single block - row_028032_col_025088.npz
# CE is the focal with gamma=0
# CE*1.0          : Epoch: 0432|Loss: 0.015551 IoU: 0.990121
# CE*1.0+tv*0.001 : Epoch: 0540|Loss: 0.009389|IoU: 0.990057
# CE*1.0+tv*0.1   : Epoch: 0555|Loss: 0.017215|IoU: 0.990685 (loss unstable)
