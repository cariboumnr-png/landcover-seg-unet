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
Base classes for UNet backbone architectures.

Provides the base class and common interface for UNet backbone
implementations, extending the abstract `Backbone` contract with
encoder/decoder methods and bottleneck infrastructure specific to
UNet-style feature extraction.
'''

# standard imports
import abc
# third-party imports
import torch
# local imports
import landseg.models.backbones as backbones
import landseg.models.backbones.unet.body.configs as body_configs
import landseg.models.backbones.unet.components as components

class UNetBackbone(backbones.Backbone):
    '''
    Abstract base class for UNet-family feature extractor backbones.

    Implementations must produce a feature map with a known channel
    width matching the prediction heads' expected input (`base_ch`).
    '''

    def __init__(
        self,
        in_ch: int,
        config: body_configs.UNetBodyConfig,
    ):
        '''
        Initialize UNet backbone encoder and bottleneck slots.

        Constructs the initial convolution block and encoder pathway.
        The bottleneck is left uninitialized and must be set separately
        via `build_bottleneck()`. Subclasses are responsible for
        instantiating the decoder pathway and implementing the abstract
        methods.

        Args:
            in_ch:
                Number of input channels from the input image or
                concatenated domain features.
            config:
                `UNetBodyConfig` specifying base channel width,
                encoder convolution parameters, and other
                architectural options.
        '''
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
        Encode input tensor into multi-scale feature maps.

        Args:
            x:
                Input tensor of shape `(B, C_in, H, W)`.

        Returns:
            tuple[torch.Tensor, ...]:
                Multi-scale encoder and bottleneck feature tensors from
                finest to coarsest resolution.

        Raises:
            NotImplementedError:
                If the subclass does not implement `encode`.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        '''
        Decode multi-scale encoder features into a feature map.

        Args:
            xs:
                Tuple of multi-scale feature tensors produced by
                `encode`.

        Returns:
            torch.Tensor:
                Decoded feature map of shape `(B, out_channels, H, W)`.

        Raises:
            NotImplementedError:
                If the subclass does not implement `decode`.
        '''
        raise NotImplementedError

    def build_bottleneck(
        self,
        config: components.BottleneckConfig,
        spatial_size: int | None = None,
    ) -> None:
        '''
        Build and attach the configured bottleneck layer.

        Args:
            config:
                `BottleneckConfig` specifying the bottleneck variant
                and layer hyperparameters.
            spatial_size:
                Spatial dimension at bottleneck resolution, required
                for transformer and hybrid bottleneck variants.

        Raises:
            ValueError:
                If `config.variant` is not a supported bottleneck type.
        '''
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
