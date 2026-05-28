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
# local imports
import landseg.models.backbones as backbones
import landseg.models.backbones.unet.components as components

class UNetBackbone(backbones.Backbone):
    '''
    Contract for any feature extractor used by MultiHeadModel.
    Implementations must produce a feature map with a known channel
    width that matches the heads' expected input (e.g., base_ch).
    '''

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        bottleneck: components.BaseBottleneck,
        **kwargs
    ) -> None:
        super().__init__()

        # initial convolution block with no norm nor drop outs
        self.inc = components.DoubleConv(in_ch, base_ch)
        # downsampling path (encoder) with 4 levels
        self.downs = components.UNetEncoders(in_ch, base_ch, **kwargs)
        # bottleneck (deepest representation) with sanity checks
        if not isinstance(bottleneck, components.BaseBottleneck):
            raise TypeError('bottleneck must be a components.BaseBottleneck')
        if self.bottleneck_ch != bottleneck.in_ch:
            raise ValueError(
                f'bottleneck_ch ({bottleneck.in_ch}) must match '
                f'encoder output channels ({self.bottleneck_ch})'
            )
        self.bottleneck = bottleneck

    @property
    @abc.abstractmethod
    def bottleneck_ch(self) -> int:
        '''Return the bottleneck channel number.'''
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W] (or possibly downsampled; the framework
                should document expectations, e.g., same spatial size if
                heads are dense).
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        '''
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W] (or possibly downsampled; the framework
                should document expectations, e.g., same spatial size if
                heads are dense).
        '''
        raise NotImplementedError
