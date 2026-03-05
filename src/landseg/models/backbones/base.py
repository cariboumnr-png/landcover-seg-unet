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

'''Base classes for architecture backbones.'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class Backbone(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Contract for any feature extractor used by MultiHeadModel.
    Implementations must produce a feature map with a known channel
    width that matches the heads' expected input (e.g., base_ch).
    '''

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        '''Number of channels in the output feature map (C_out).'''
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W] (or possibly downsampled; the framework
                should document expectations, e.g., same spatial size if
                heads are dense).
        '''
        raise NotImplementedError
