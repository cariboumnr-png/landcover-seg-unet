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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Composite loss manager for combining multiple primitive loss functions.

This module provides a flexible interface for constructing weighted
combinations of primitive loss types (e.g., focal, dice) based on a user
configuration. Each component loss must implement the PrimitiveLoss
interface and is instantiated from a small registry of supported types.

The main entry point is `CompositeLoss`, which handles:
    - Instantiating individual loss modules,
    - Managing their per-component weights,
    - Computing a weighted sum of all enabled losses
'''

# standard typing
from __future__ import annotations
import typing
# third-party imports
import torch
import torch.nn
# local imports
import landseg.trainer.components.task.loss as loss

# ---------------------------------Public Type---------------------------------
class LossTypes(typing.Protocol):
    @property
    def focal(self) -> _FocalConfig:...
    @property
    def dice(self) -> _DiceConfig:...

# --------------------------------private  type--------------------------------
class _FocalConfig(typing.Protocol):
    @property
    def weight(self) -> float:...
    @property
    def gamma(self) -> float:...
    @property
    def reduction(self) -> str:...

class _DiceConfig(typing.Protocol):
    @property
    def weight(self) -> float:...
    @property
    def smooth(self) -> float:...

# --------------------------------Public  Class--------------------------------
class CompositeLoss(torch.nn.Module):
    '''
    Combine multiple loss components into a weighted composite loss.

    Supported loss types are defined in the class-level `registry`.
    Configuration is provided as a dictionary describing which loss
    components to enable and the parameters for each.

    Example:
        Base configuration enabling focal + dice losses:

            sample_config = {
                "focal": {
                    "weight": 0.7,
                    "alpha": None,
                    "gamma": 2.0,
                    "reduction": "mean",
                },
                "dice": {
                    "weight": 0.3,
                    "smooth": 1.0,
                }
            }

        Create a composite loss manually:
        >>> from landseg.training import loss
        >>> sample_class = loss.CompositeLoss(sample_config, 255)

        When used via `build_headlosses`, per-head a values for
        focal loss are injected dynamically from head specifications.

    Each enabled component is instantiated, stored in a ModuleList, and
    evaluated during the forward pass.
    '''

    def __init__(
        self,
        config: LossTypes,
        ignore_index: int,
        alpha: list[float] | None = None
    ):
        '''
        Initialize the composite loss from a configuration dictionary.

        Creates:
            - self.losses: a ModuleList of instantiated loss functions.
            - self.weights: parallel list of scalar weights for each
              component loss.

        Args:
            config: Dictionary mapping loss-type names to their parameter
                blocks. Must satisfy `loss.is_loss_types(...)`.
            ignore_index: Label index to ignore in all component losses
                that support masking of invalid or void labels.
        '''

        super().__init__()

        # expose ignore index
        self.ignore_index = ignore_index

        # iterate through currently support loss types
        self.losses = torch.nn.ModuleList()
        self.weights: list[float] = []

        # focal loss
        if config.focal.weight:
            loss_fn = loss.FocalLoss(
                alpha=alpha,
                gamma=config.focal.gamma,
                reduction=config.focal.reduction,
                ignore_index=ignore_index
            )
            self.losses.append(loss_fn)
            self.weights.append(config.focal.weight)

        # dice loss
        if config.dice.weight:
            loss_fn = loss.DiceLoss(
                smooth=config.dice.smooth,
                ignore_index=ignore_index
            )
            # add to sequences
            self.losses.append(loss_fn)
            self.weights.append(config.dice.weight)

    def forward(
        self,
        p: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        '''
        Compute the weighted sum of all enabled loss components.

        Args:
            p: Model logits or predictions.
            t: Target labels with a shape compatible with `p`.
            masks: Optional dictionary of masks passed to each loss
                component (keyword-only argument).

        Returns:
            A scalar tensor representing the total loss across all
            components. NaN/Inf propagation is controlled by relying on
            each primitive loss implementation.
        '''

        # get mask
        masks = kwargs.get('masks', None)
        # accumulate included losses
        total_loss = p.new_zeros(())
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_val = loss_fn(p, t, masks=masks)
            total_loss += weight * loss_val
        return total_loss
