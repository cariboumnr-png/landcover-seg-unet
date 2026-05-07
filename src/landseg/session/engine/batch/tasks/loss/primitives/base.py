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
Base classes for primitive loss components.

Defines an abstract interface for loss modules operating on model logits,
targets, and optional per-pixel or per-class masks. Concrete loss
implementations should inherit from `PrimitiveLoss` and implement
`forward()`.
'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

# --------------------------------Public  Class--------------------------------
class PrimitiveLoss(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract base class for loss computation modules.

    Subclasses must implement `forward()` and return a scalar loss
    tensor. The interface supports optional mask dictionaries for
    selective weighting of pixels or classes.
    '''
    @abc.abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        masks: dict[float, torch.Tensor] | None,
        features: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Compute the loss from model logits and targets.

        Args:
            logits: Prediction logits of shape [...], typically
                (B, C, H, W) or (B, C).
            targets: Ground-truth labels with a shape compatible with
                logits.
            masks: Optional mapping from weights (floats) to boolean or
                float masks of the same spatial shape as targets, used
                to apply selective weighting.
            features: Optional feature tensor (e.g., encoder output or
                input image) of shape (B, D, H, W), used by certain
                losses (e.g., smoothness, contrastive).

        Returns:
            A scalar tensor representing the computed loss.
        '''

    @staticmethod
    def _compose_pixel_weights(
        *,
        masks: dict[float, torch.Tensor] | None,
        targets: torch.Tensor,
        ignore_index: int | None,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        '''
        Construct a per-pixel weight map in the range [0, 1].

        The resulting tensor assigns a weight to each pixel in `targets`,
        which can be used to scale loss contributions during training.

        Processing steps:
        1. Initialize all pixel weights to 1.0.
        2. If `masks` is provided, iteratively apply each mask:
            - Each key-value pair (w_i, mask_i) represents a
                down-weighting rule.
            - For pixels where `mask_i` is True:
                weight = min(current_weight, clamp(w_i, 0, 1))
            - This ensures weights can only decrease (never increase).
        3. If `ignore_index` is specified:
            - Pixels in `targets` equal to `ignore_index` assigned
                to weight 0.

        Args:
            masks:
                Mapping of scalar weights to boolean-compatible masks.
                - Keys: Desired weight values (will clampe to [0, 1]).
                - Values: Tensors with the same shape as `targets` (which
                    pixels the weight should apply to).
                - If None, no mask-based weighting is applied.

            targets:
                Tensor containing target labels. Defines the shape of the
                output weight map.

            ignore_index:
                Target value to ignore. Pixels with this value will
                receive a weight of 0. If None, no pixels are ignored.

            device:
                Device on which the output tensor will be allocated.

            dtype:
                Data type of the output weight tensor.

        Returns:
            A tensor of the same shape as `targets`, with values in
            [0, 1], representing per-pixel weights.

        Raises:
            AssertionError:
                If mask keys are not numeric / mask tensors have
                incompatible shapes.
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
