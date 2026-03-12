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

        Returns:
            A scalar tensor representing the computed loss.
        '''
