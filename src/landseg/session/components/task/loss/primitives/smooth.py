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

import torch
import torch.nn.functional
import landseg.session.components.task.loss.primitives as primitives


class SpectralSmoothnessLoss(primitives.PrimitiveLoss):
    '''
    Pairwise spectral smoothness regularizer.

    The loss penalizes prediction differences between neighboring pixels,
    weighted by feature similarity. Optional masks are converted into a
    per-pixel weight map and then applied pairwise.
    '''

    def __init__(
        self,
        ignore_index: int,
        *,
        alpha: float,
        neighbour: int = 4,
    ) -> None:
        '''
        Initialize the smoothness loss.

        Args:
            ignore_index:
                Label to exclude entirely from the loss.
            alpha:
                Controls the sharpness of the feature-similarity weight.
            neighbour:
                Neighborhood connectivity, either 4 or 8.
        '''
        super().__init__()

        self.ignore_index = ignore_index
        self.alpha = alpha
        # neighbourhood
        match neighbour:
            case 4: self.offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            case 8: self.offsets = [(0, 1), (1, 0), (0, -1), (-1, 0),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
            case _: raise ValueError('Neighbourhood must be 4 or 8.')

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        masks: dict[float, torch.Tensor] | None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''
        Compute the spectral smoothness loss.

        Args:
            logits:
                Tensor of shape (B, C, H, W), raw class logits.
            targets:
                Tensor of shape (B, H, W), integer class labels.
            masks:
                Optional mapping from scalar weights to boolean masks.
            features:
                Tensor of shape (B, D, H, W), used to derive pairwise
                feature similarity weights.

        Returns:
            A scalar loss tensor.
        '''

        # Normalize features along channel dimension.
        features = self._is_valid_inputs(features, logits, targets)
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Convert logits to probabilities.
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Compose per-pixel weights from masks and ignore_index.
        pixel_weights = self._compose_pixel_weights(
            masks=masks,
            targets=targets,
            ignore_index=self.ignore_index,
            device=logits.device,
            dtype=logits.dtype,
        )

        feat_shift_list: list[torch.Tensor] = []
        prob_shift_list: list[torch.Tensor] = []
        weight_shift_list: list[torch.Tensor] = []
        valid_mask_list: list[torch.Tensor] = []

        # logits: [B, C, H, W]
        _, _, height, width = logits.shape

        for dy, dx in self.offsets:
            feat_shift_list.append(
                torch.roll(features, shifts=(dy, dx), dims=(2, 3))
            )
            prob_shift_list.append(
                torch.roll(probs, shifts=(dy, dx), dims=(2, 3))
            )
            weight_shift_list.append(
                torch.roll(pixel_weights, shifts=(dy, dx), dims=(1, 2))
            )

            valid = torch.ones(
                (height, width),
                device=logits.device,
                dtype=logits.dtype,
            )

            if dy > 0:
                valid[:dy, :] = 0
            elif dy < 0:
                valid[dy:, :] = 0

            if dx > 0:
                valid[:, :dx] = 0
            elif dx < 0:
                valid[:, dx:] = 0

            valid_mask_list.append(valid)

        # Stack shifted tensors across the neighbour dimension N.
        feat_shift = torch.stack(feat_shift_list, dim=1)
        prob_shift = torch.stack(prob_shift_list, dim=1)
        weight_shift = torch.stack(weight_shift_list, dim=1).unsqueeze(2)

        # Shape:
        #   features_center -> (B, 1, D, H, W)
        #   probs_center    -> (B, 1, C, H, W)
        #   weight_center   -> (B, 1, 1, H, W)
        features_center = features.unsqueeze(1)
        probs_center = probs.unsqueeze(1)
        weight_center = pixel_weights.unsqueeze(1).unsqueeze(2)

        # Shape: (1, N, 1, H, W)
        valid_mask = torch.stack(valid_mask_list, dim=0)
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(2)

        # Pairwise feature distance and prediction distance.
        dist_x = ((features_center - feat_shift) ** 2).sum(
            dim=2,
            keepdim=True,
        )
        dist_p = ((probs_center - prob_shift) ** 2).sum(
            dim=2,
            keepdim=True,
        )

        # Feature similarity weight.
        spectral_weight = torch.exp(-self.alpha * dist_x)

        # Pairwise pixel weight:
        # - respects ignore_index and upstream mask values,
        # - zeroes a pair if either endpoint is invalid,
        # - preserves your min-based down-weighting semantics.
        pair_weight = torch.minimum(weight_center, weight_shift) * valid_mask

        weighted_loss = spectral_weight * dist_p * pair_weight

        eps = torch.finfo(logits.dtype).eps
        denom = pair_weight.sum().clamp_min(eps)

        return weighted_loss.sum() / denom

    @staticmethod
    def _is_valid_inputs(
        features: torch.Tensor | None,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        if features is None:
            raise ValueError('Features are required for smoothness loss.')

        if logits.ndim != 4:
            raise ValueError(
                f'Expected logits with shape (B, C, H, W), got {logits.shape}.'
            )

        if targets.ndim != 3:
            raise ValueError(
                f'Expected targets with shape (B, H, W), got {targets.shape}.'
            )

        if features.ndim != 4:
            raise ValueError(
                f'Expected features with shape (B, D, H, W), '
                f'got {features.shape}.'
            )

        if logits.shape[0] != targets.shape[0]:
            raise ValueError('Batch size mismatch between logits and targets.')

        if logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError(
                'Spatial size mismatch between logits and targets.'
            )

        if features.shape[0] != logits.shape[0]:
            raise ValueError('Batch size mismatch between features and logits.')

        if features.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                'Spatial size mismatch between features and logits.'
            )

        return features
