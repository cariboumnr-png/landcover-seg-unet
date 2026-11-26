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

'''Model backbone and bottleneck configuration schemas.'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base


# alias
field = dataclasses.field


# ----- shared double convolution
@dataclasses.dataclass
class ConvParams(base.BaseConfigSection):
    '''Shared double-convolution block configuration.'''
    norm: str | None = 'gn'
    gn_groups: int | None = 8
    p_drop: float = 0.0


# ----- UNet bodies
@dataclasses.dataclass
class UNetBodyConfig(base.BaseConfigSection):
    '''Base configuration for UNet-family encoder-decoder bodies.'''
    body: str
    base_ch: int
    encoder_conv_params: ConvParams
    nodes_conv_params: ConvParams | None
    decoder_conv_params: ConvParams | None


@dataclasses.dataclass
class UNet(UNetBodyConfig):
    '''Standard UNet architecture body configuration.'''
    body: str = 'unet'
    base_ch: int = 32
    encoder_conv_params: ConvParams = field(default_factory=ConvParams)
    nodes_conv_params: ConvParams | None = None
    decoder_conv_params: ConvParams | None = field(default_factory=ConvParams)


@dataclasses.dataclass
class UNetPP(UNetBodyConfig):
    '''UNet++ nested architecture body configuration.'''
    body: str = 'unetpp'
    base_ch: int = 32
    encoder_conv_params: ConvParams = field(default_factory=ConvParams)
    nodes_conv_params: ConvParams | None = field(default_factory=ConvParams)
    decoder_conv_params: ConvParams | None = None


@dataclasses.dataclass
class UNetPPP(UNetBodyConfig):
    '''UNet 3+ full-scale skip architecture body configuration.'''
    body: str = 'unetppp'
    base_ch: int = 32
    encoder_conv_params: ConvParams = field(default_factory=ConvParams)
    nodes_conv_params: ConvParams | None = field(default_factory=ConvParams)
    decoder_conv_params: ConvParams | None = None


# ----- UNet bottlenecks
@dataclasses.dataclass
class Transformer:
    '''Transformer block hyperparameters for bottleneck modules.'''
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attn_dropout: float = 0.0

@dataclasses.dataclass
class BottleneckConfig:
    '''Base configuration for UNet bottleneck blocks.'''
    variant: str
    num_conv_blocks: int | None
    conv_params: ConvParams | None
    num_transformer_blocks: int | None
    transformer_params: Transformer | None

@dataclasses.dataclass
class UNetBottleneckConfig(BottleneckConfig):
    '''Convolutional UNet bottleneck configuration.'''
    variant: str = 'conv'
    num_conv_blocks: int | None = None
    conv_params: ConvParams | None = field(default_factory=ConvParams)
    num_transformer_blocks: int | None = None
    transformer_params: Transformer | None = None

@dataclasses.dataclass
class TransformerBottleneckConfig(BottleneckConfig):
    '''Transformer-based UNet bottleneck configuration.'''
    variant: str = 'transformer'
    num_conv_blocks: int | None = None
    conv_params: ConvParams | None = None
    num_transformer_blocks: int | None = 4
    transformer_params: Transformer | None = field(default_factory=Transformer)

@dataclasses.dataclass
class HybridBottleneckConfig(BottleneckConfig):
    '''Hybrid convolution and transformer bottleneck configuration.'''
    variant: str = 'hybrid'
    num_conv_blocks: int | None = 2
    conv_params: ConvParams | None = field(default_factory=ConvParams)
    num_transformer_blocks: int | None = 2
    transformer_params: Transformer | None = field(default_factory=Transformer)


@dataclasses.dataclass
class UNetBackboneConfig:
    '''Composite UNet body and bottleneck configuration.'''
    body: UNetBodyConfig
    bottleneck: BottleneckConfig


def default_bodies() -> dict[str, typing.Any]:
    '''
    Build default UNet body configuration registry.

    Returns:
        dict[str, typing.Any]:
            Mapping from body variant names to default configs.
    '''
    return {
        'unet': UNet(),
        'unetpp': UNetPP(),
        'unetppp': UNetPPP(),
    }


def default_bottlenecks() -> dict[str, typing.Any]:
    '''
    Build default UNet bottleneck configuration registry.

    Returns:
        dict[str, typing.Any]:
            Mapping from bottleneck variant names to default configs.
    '''
    return {
        'conv': UNetBottleneckConfig(),
        'transformer': TransformerBottleneckConfig(),
        'hybrid': HybridBottleneckConfig(),
    }
