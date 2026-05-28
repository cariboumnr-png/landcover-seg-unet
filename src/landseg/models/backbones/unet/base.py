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

# pylint: disable=missing-function-docstring

'''Base classes for architecture backbones.'''

# standard imports
import abc
# third-party imports
import torch
# local imports
import landseg.models.backbones as backbones
import landseg.models.backbones.unet as unet
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
        config: unet.BackboneConfig,
    ):
        '''doc'''

        super().__init__()
        base_ch = config.base_ch
        enc_conv_params = config.encoder_conv_params
        # initial convolution block with no norm nor drop outs
        self.inc = components.DoubleConv(in_ch, base_ch, None)
        # downsampling path (encoder) with 4 levels
        self.downs = components.UNetEncoders(in_ch, base_ch, enc_conv_params)
        # bottleneck type declaration
        self.bottleneck: components.BaseBottleneck

    @property
    @abc.abstractmethod
    def bottleneck_ch(self) -> int:
        '''Return the expected bottleneck channel number.'''
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

    def build_bottleneck(
        self,
        config: components.BottleneckConfig,
        spatial_size: int | None = None,
    ) -> None:
        '''Build bottleneck layer'''

        match config.variant:
            case 'conv':
                assert config.conv_params
                self.bottleneck = components.UNetBottleneck(
                    self.bottleneck_ch,
                    config.conv_params
                )
            case 'transformer':
                assert spatial_size is not None
                self.bottleneck = components.TransformerBottleneck(
                    self.bottleneck_ch,
                    spatial_size,
                    config
                )
            case 'hybrid':
                assert spatial_size is not None
                self.bottleneck = components.HybridBottleneck(
                    self.bottleneck_ch,
                    spatial_size,
                    config
            )
            case _: raise ValueError('Invalid bottleneck variant')
