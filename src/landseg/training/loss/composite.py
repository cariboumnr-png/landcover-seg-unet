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

'''Composite loss calculation manager.'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.training.loss as loss

class CompositeLoss(torch.nn.Module):
    '''doc'''

    # class-level flexible registry of supported loss types
    registry = {
            "focal": loss.FocalLoss,
            "dice": loss.DiceLoss,
    }

    def __init__(
            self,
            config: dict,
            ignore_index: int
        ):
        '''
        Args:
            config (dict): Specifies loss types and their parameters.
        '''

        super().__init__()
        # type check input config
        if loss.is_loss_types(config):
            self.cfgs = config
        else:
            raise ValueError('Input config not compliant with `LossTypes`')

        # make ignore index public
        self.ignore_index = ignore_index

        # iterate through input loss types and gather corresponding blocks
        self.losses = torch.nn.ModuleList()
        self.weights: list[float] = []
        # loss function by type
        # focal
        focal = self.cfgs.get('focal', None)
        if focal is not None:
            weight = focal['weight']
            loss_fn = self.registry['focal'](
                alpha=focal['alpha'],
                gamma=focal['gamma'],
                reduction=focal['reduction'],
                ignore_index=ignore_index
            )
            # add to sequences
            self.losses.append(loss_fn)
            self.weights.append(weight)
        # dice
        dice = self.cfgs.get('dice', None)
        if dice is not None:
            weight = dice['weight']
            loss_fn = self.registry['dice'](
                smooth=dice['smooth'],
                ignore_index=ignore_index
                )
            # add to sequences
            self.losses.append(loss_fn)
            self.weights.append(weight)

    def forward(
            self,
            p: torch.Tensor,
            t: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
        '''Compute combined loss with NaN/Inf check before output.'''

        # get mask
        masks = kwargs.get('masks', None)
        # accumulate included losses
        total_loss = p.new_zeros(())
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_val = loss_fn(p, t, masks=masks)
            total_loss += weight * loss_val
        return total_loss
