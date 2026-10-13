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

# pylint: disable=missing-class-docstring

'''Model architecture schema. Field validation pending'''

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
    norm: str | None = 'gn'
    gn_groups: int | None = 8
    p_drop: float = 0.0


# ----- UNet bodies
@dataclasses.dataclass
class UNetBodyConfig(base.BaseConfigSection):
    body: str
    base_ch: int
    encoder_conv_params: ConvParams
    nodes_conv_params: ConvParams | None
    decoder_conv_params: ConvParams | None


@dataclasses.dataclass
class UNet(UNetBodyConfig):
    body: str = 'unet'
    base_ch: int = 32
    encoder_conv_params: ConvParams = field(default_factory=ConvParams)
    nodes_conv_params: ConvParams | None = None
    decoder_conv_params: ConvParams | None = field(default_factory=ConvParams)


@dataclasses.dataclass
class UNetPP(UNetBodyConfig):
    body: str = 'unetpp'
    base_ch: int = 32
    encoder_conv_params: ConvParams = field(default_factory=ConvParams)
    nodes_conv_params: ConvParams | None = field(default_factory=ConvParams)
    decoder_conv_params: ConvParams | None = None


@dataclasses.dataclass
class UNetPPP(UNetBodyConfig):
    body: str = 'unetppp'
    base_ch: int = 32
    encoder_conv_params: ConvParams = field(default_factory=ConvParams)
    nodes_conv_params: ConvParams | None = field(default_factory=ConvParams)
    decoder_conv_params: ConvParams | None = None


# ----- UNet bottlenecks
@dataclasses.dataclass
class Transformer:
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attn_dropout: float = 0.0

@dataclasses.dataclass
class BottleneckConfig:
    variant: str
    num_conv_blocks: int | None
    conv_params: ConvParams | None
    num_transformer_blocks: int | None
    transformer_params: Transformer | None

@dataclasses.dataclass
class UNetBottleneckConfig(BottleneckConfig):
    variant: str = 'conv'
    num_conv_blocks: int | None = None
    conv_params: ConvParams | None = field(default_factory=ConvParams)
    num_transformer_blocks: int | None = None
    transformer_params: Transformer | None = None

@dataclasses.dataclass
class TransformerBottleneckConfig(BottleneckConfig):
    variant: str = 'transformer'
    num_conv_blocks: int | None = None
    conv_params: ConvParams | None = None
    num_transformer_blocks: int | None = 4
    transformer_params: Transformer | None = field(default_factory=Transformer)

@dataclasses.dataclass
class HybridBottleneckConfig(BottleneckConfig):
    variant: str = 'hybrid'
    num_conv_blocks: int | None = 2
    conv_params: ConvParams | None = field(default_factory=ConvParams)
    num_transformer_blocks: int | None = 2
    transformer_params: Transformer | None = field(default_factory=Transformer)


@dataclasses.dataclass
class UNetBackboneConfig:
    body: UNetBodyConfig
    bottleneck: BottleneckConfig


def default_bodies() -> dict[str, typing.Any]:
    '''Return default UNet body registry.'''
    return {
        'unet': UNet(),
        'unetpp': UNetPP(),
        'unetppp': UNetPPP(),
    }


def default_bottlenecks() -> dict[str, typing.Any]:
    '''Return default UNet bottleneck registry.'''
    return {
        'conv': UNetBottleneckConfig(),
        'transformer': TransformerBottleneckConfig(),
        'hybrid': HybridBottleneckConfig(),
    }
