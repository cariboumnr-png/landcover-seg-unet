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
Spectral smoothness loss for segmentation tasks.

Encourages neighboring pixels with similar features to produce consistent
predictions, promoting spatial coherence in outputs.

This module defines a single loss primitive used by higher-level loss
composition components in the execution pipeline.
'''


# third-party imports
import torch
import torch.nn.functional
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives as primitives

class SpectralSmoothnessLoss(primitives.PrimitiveLoss):
    '''
    Pairwise spectral smoothness regularizer.

    The loss penalizes prediction differences between neighboring pixels,
    weighted by feature similarity. Optional masks are converted into a
    per-pixel weight map and then applied pairwise.

    The goal is to encourage neighboring pixels with similar spectral
    signals to have similar predicted class probabilities.

    For each pixel i and each spatial neighbor j, the loss computes:

        s(i, j) = exp(-alpha * ||f_i - f_j||^2) # similarity
        d(i, j) = ||p_i - p_j||^2               # disagreement

    where:
        - f_i and f_j are the feature vectors at pixels i and j
        - p_i and p_j are the softmax class-probability vectors at pixels
            i and j
        - alpha controls how quickly the similarity weight decays with
        feature distance

    The total loss is the weighted average of pairwise probability
    disagreement over valid neighboring pixel pairs:

        L_smooth = sum_ij [w_ij * s(i, j) * d(i, j)] / sum_ij [w_ij]

    where w_ij is the pairwise mask/weight for the pixel pair and invalid
    border pairs are excluded.

    Note:
    - Lower alpha makes the similarity weight stay closer to 1, forcing
      aggressive smoothing across different-looking pixels;
    - higher alpha makes the weight decay rapidly, restricting smoothing
      to only nearly identical spectral signatures.
    '''

    def __init__(
        self,
        *,
        alpha: float,
        neighbour: int,
        spectral_bands: list[int] | None,
        ignore_index: int,
    ) -> None:
        '''
        Initialize the smoothness loss.

        Args:
            alpha:
                Controls the sharpness of the feature-similarity weight.
                Lower alpha = stronger/higher penalty for predicting
                different classes over similar pixels.
            neighbour:
                Neighborhood connectivity, either 4 or 8.
            spectral_bands:
                0-based indices mapping logits channels as spectral
            ignore_index:
                Label to exclude entirely from the loss.
        '''
        super().__init__()

        self.alpha = alpha
        self.ignore_index = ignore_index

        match neighbour:
            case 4: self.offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            case 8: self.offsets = [(0, 1), (1, 0), (0, -1), (-1, 0),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
            case _: raise ValueError('Neighbourhood must be 4 or 8.')

        if spectral_bands:
            self.spec_idx = torch.as_tensor(spectral_bands, dtype=torch.long)
        else:
            self.spec_idx = None

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
        # validate input tensors where features are needed
        if features is None:
            raise ValueError('Features are required for smoothness loss')
        self._validate_inputs(logits, targets, features)

        # prepare inputs
        features = self._prepare_features(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        weights = self._compose_pixel_weights(
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

        for offset in self.offsets: # dy, dx
            feat_shift_list.append(torch.roll(features, offset, dims=(2, 3)))
            prob_shift_list.append(torch.roll(probs, offset, dims=(2, 3)))
            weight_shift_list.append(torch.roll(weights, offset, dims=(1, 2)))
            valid_mask_list.append(
                self._valid_neighbors(
                    (logits.shape[2], logits.shape[3]), # [_, _, H, W]
                    offset,
                    dtype=logits.dtype,
                    device=logits.device
                )
            )

        # stack shifted tensors across the neighbour dimension N.
        feat_shift = torch.stack(feat_shift_list, dim=1)
        prob_shift = torch.stack(prob_shift_list, dim=1)
        weight_shift = torch.stack(weight_shift_list, dim=1).unsqueeze(2)
        # stack valid masks: (1, N, 1, H, W)
        mask = torch.stack(valid_mask_list, dim=0).unsqueeze(0).unsqueeze(2)

        # Shape:
        #   features_center -> (B, 1, D, H, W)
        #   probs_center    -> (B, 1, C, H, W)
        #   weight_center   -> (B, 1, 1, H, W)
        feat_centre = features.unsqueeze(1)
        probs_centre = probs.unsqueeze(1)
        weight_centre = weights.unsqueeze(1).unsqueeze(2)


        # Pairwise feature distance and prediction distance.
        dist_x = ((feat_centre - feat_shift) ** 2).sum(dim=2, keepdim=True)
        dist_p = ((probs_centre - prob_shift) ** 2).sum(dim=2, keepdim=True)

        # Feature similarity weight.
        spectral_weight = torch.exp(-self.alpha * dist_x)

        # Pairwise pixel weight:
        # - respects ignore_index and upstream mask values,
        # - zeroes a pair if either endpoint is invalid,
        # - preserves your min-based down-weighting semantics.
        pair_weight = torch.minimum(weight_centre, weight_shift) * mask
        weighted_loss = spectral_weight * dist_p * pair_weight

        # safe division and return
        eps = torch.finfo(logits.dtype).eps
        return weighted_loss.sum() / pair_weight.sum().clamp_min(eps)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        '''Filter and normalize features.'''
        # filter feature channels if indices provided
        if self.spec_idx is not None:
            self.spec_idx = self.spec_idx.to(features.device)
            features = features.index_select(1, self.spec_idx)
        # normalize features along channel dimension.
        features = torch.nn.functional.normalize(features, p=2, dim=1) # L2
        return features

    @staticmethod
    def _valid_neighbors(
        size: tuple[int, int],
        offsets: tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device
    ):
        '''Return mask of valid neighbouring pixels.'''
        valid = torch.ones(size, dtype=dtype, device=device,)
        dy, dx = offsets

        if dy > 0:
            valid[:dy, :] = 0
        elif dy < 0:
            valid[dy:, :] = 0

        if dx > 0:
            valid[:, :dx] = 0
        elif dx < 0:
            valid[:, dx:] = 0

        return valid
