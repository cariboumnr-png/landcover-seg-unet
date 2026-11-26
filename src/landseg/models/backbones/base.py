# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Base classes for architecture backbones.

This module defines the abstract interface for all backbone feature
extractors used by multi-head segmentation models. Implementations must
specify their output channel width and spatial divisor constraints to
enable downstream validation and conditioning integration.
'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class Backbone(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract base contract for segmentation model feature backbones.

    Implementations must produce a feature map with a known channel
    width matching the prediction heads' expected input (`base_ch`).
    '''

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        '''Return number of channels in the output feature map.'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def spatial_divisor(self) -> int:
        '''Return integer divisor constraining spatial dimensions.'''
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute output feature map from an input tensor.

        Args:
            x:
                Input tensor of shape `(B, C_in, H, W)`.

        Returns:
            torch.Tensor:
                Extracted feature map of shape `(B, C_out, H, W)`.

        Raises:
            NotImplementedError:
                If the subclass does not implement `forward`.
        '''
        raise NotImplementedError
