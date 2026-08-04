# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Ecological similarity regularization loss for species segmentation.

Provides a domain-aware regularizer that penalizes prediction probabilities
proportional to the ecological dissimilarity between target and predicted
species classes using a precomputed cosine similarity matrix.
'''

# third-party imports
import torch
import torch.nn
import torch.nn.functional
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives as primitives


# --------------------------------Public  Class--------------------------------
class EcologicalSimilarityLoss(primitives.PrimitiveLoss):
    '''
    Ecological similarity regularization loss primitive.

    Penalizes predicted class probabilities based on a precomputed cosine
    dissimilarity matrix (1 - S), where S is the N x N species similarity
    matrix.
    '''

    similarity_matrix: torch.Tensor

    def __init__(
        self,
        *,
        similarity_matrix: torch.Tensor,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ) -> None:
        '''
        Initialize an ecological similarity regularization module.

        Args:
            similarity_matrix: Precomputed N x N species cosine similarity
                torch tensor.
            ignore_index: Target label value to ignore in loss computation.
            reduction: Reduction method ('mean', 'sum', 'none').
        '''
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer('similarity_matrix', similarity_matrix)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        masks: dict[float, torch.Tensor] | None = None,
        features: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Compute ecological dissimilarity regularization loss.

        Args:
            logits: Model predictions tensor of shape (B, C, H, W).
            targets: Target label indices tensor of shape (B, H, W).
            masks: Optional dictionary of per-pixel weight masks.
            features: Optional bottleneck features (unused).

        Returns:
            Scalar loss tensor.
        '''
        # compute softmax probabilities along channel dimension
        probs = torch.nn.functional.softmax(logits, dim=1)

        # create valid target mask excluding ignore_index
        valid_mask = targets != self.ignore_index
        clamped_targets = torch.where(valid_mask, targets, 0)

        # construct dissimilarity matrix D = 1.0 - S
        sim_mat = self.similarity_matrix.to(
            device=logits.device, dtype=logits.dtype
        )
        dissimilarity_matrix = 1.0 - sim_mat

        # gather per-pixel dissimilarity vector for target class y
        target_dissimilarity = dissimilarity_matrix[clamped_targets].permute(
            0, 3, 1, 2
        )

        # per-pixel ecological loss: sum_c p_c * (1 - S_{y, c})
        pixel_loss = torch.sum(probs * target_dissimilarity, dim=1)

        # zero out ignored pixels
        pixel_loss = torch.where(valid_mask, pixel_loss, 0.0)

        # apply optional mask dictionary weighting if provided
        if masks is not None:
            for weight, mask in masks.items():
                pixel_loss = pixel_loss * (1.0 + (weight - 1.0) * mask)

        if self.reduction == 'mean':
            num_valid = valid_mask.sum().clamp(min=1)
            return pixel_loss.sum() / num_valid
        if self.reduction == 'sum':
            return pixel_loss.sum()
        return pixel_loss
