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
# pylint: disable=too-few-public-methods

'''
Model architecture schema
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing

# alias
field = dataclasses.field

# ---------------------------------MODELS CONFIGS------------------------------
# ----- shared double convolution
@dataclasses.dataclass
class _ConvParams:
    norm: str | None = 'gn'
    gn_groups: int | None = 8
    p_drop: float = 0.0

# ----- UNet bodies
@dataclasses.dataclass
class _UNetBodyConfig:
    body: str
    base_ch: int
    encoder_conv_params: _ConvParams
    nodes_conv_params: _ConvParams | None
    decoder_conv_params: _ConvParams | None

@dataclasses.dataclass
class _UNet(_UNetBodyConfig):
    body: str = 'unet'
    base_ch: int = 32
    encoder_conv_params: _ConvParams = field(default_factory=_ConvParams)
    nodes_conv_params: _ConvParams | None = None
    decoder_conv_params: _ConvParams | None = field(default_factory=_ConvParams)

@dataclasses.dataclass
class _UNetPP(_UNetBodyConfig):
    body: str = 'unetpp'
    base_ch: int = 32
    encoder_conv_params: _ConvParams = field(default_factory=_ConvParams)
    nodes_conv_params: _ConvParams | None = field(default_factory=_ConvParams)
    decoder_conv_params: _ConvParams | None = None

@dataclasses.dataclass
class _UNetPPP(_UNetBodyConfig):
    body: str = 'unetppp'
    base_ch: int = 32
    encoder_conv_params: _ConvParams = field(default_factory=_ConvParams)
    nodes_conv_params: _ConvParams | None = field(default_factory=_ConvParams)
    decoder_conv_params: _ConvParams | None = None

# ----- UNet bottlenecks
@dataclasses.dataclass
class _Transformer:
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attn_dropout: float = 0.0

@dataclasses.dataclass
class _BottleneckConfig:
    variant: str
    num_conv_blocks: int | None
    conv_params: _ConvParams | None
    num_transformer_blocks: int | None
    transformer_params: _Transformer | None

@dataclasses.dataclass
class _UNetBottleneckConfig(_BottleneckConfig):
    variant: str = 'conv'
    num_conv_blocks: int | None = None
    conv_params: _ConvParams | None = field(default_factory=_ConvParams)
    num_transformer_blocks: int | None = None
    transformer_params: _Transformer | None = None

@dataclasses.dataclass
class _TransformerBottleneckConfig(_BottleneckConfig):
    variant: str = 'transformer'
    num_conv_blocks: int | None = None
    conv_params: _ConvParams | None = None
    num_transformer_blocks: int | None = 4
    transformer_params: _Transformer | None = field(default_factory=_Transformer)

@dataclasses.dataclass
class _HybridBottleneckConfig(_BottleneckConfig):
    variant: str = 'hybrid'
    num_conv_blocks: int | None = 2
    conv_params: _ConvParams | None = field(default_factory=_ConvParams)
    num_transformer_blocks: int | None = 2
    transformer_params: _Transformer | None = field(default_factory=_Transformer)

# ----- unet backbone config
@dataclasses.dataclass
class _UNetBackboneConfig:
    body: _UNetBodyConfig
    bottleneck: _BottleneckConfig

# ----- conditioners
class _DomainTargetConfig(typing.Protocol):
    name: str
    use_ids: bool
    use_vec: bool
    ids_embd_dims: int
    vec_proj_dims: int
    vec_proj_config: _DomainProjectionConfig
    conditioner_config: _DomainConditionerAdapterConfig

class _DomainProjectionConfig(typing.TypedDict):
    use_mlp: typing.NotRequired[bool]
    hidden_dim: typing.NotRequired[int | None]
    num_hidden_layers: typing.NotRequired[int]
    dropout: typing.NotRequired[float]
    activation: typing.NotRequired[str]

class _DomainConditionerAdapterConfig(typing.TypedDict):
    hidden_dim: typing.NotRequired[int] # currently for FiLM

class _DomainTargetConfigBase(_DomainTargetConfig):
    def __post_init__(self):
        if self.use_ids and not self.ids_embd_dims:
            raise ValueError(...)
        if self.use_vec and not self.vec_proj_dims:
            raise ValueError(...)

@dataclasses.dataclass
class _Concat(_DomainTargetConfigBase):
    name: str = 'concat'
    use_ids: bool = True
    use_vec: bool = True
    ids_embd_dims: int = 4
    vec_proj_dims: int = 4
    vec_proj_config: _DomainProjectionConfig = field(
        default_factory=lambda: {
            'use_mlp': False
        }
    )
    conditioner_config: _DomainConditionerAdapterConfig = field(
        default_factory=lambda: {}
    )

@dataclasses.dataclass
class _FiLM(_DomainTargetConfigBase):
    name: str = 'film'
    use_ids: bool = True
    use_vec: bool = True
    ids_embd_dims: int = 4
    vec_proj_dims: int = 4
    vec_proj_config: _DomainProjectionConfig = field(
        default_factory=lambda: {
            'use_mlp': True,
            'hidden_dim': 128,
            'num_hidden_layers': 1,
            'activation': 'gelu',
            'dropout': 0.1,
        }
    )
    conditioner_config: _DomainConditionerAdapterConfig = field(
        default_factory=lambda: {
            'hidden_dim': 128
        }
    )

# ----- numeric safty
@dataclasses.dataclass
class _NumericSafety:
    enable_clamp: bool = True
    clamp_range: tuple[float, float] = (1e-4, 1e4)

# ----- MODELS
@dataclasses.dataclass
class ModelsConfig:
    model_body: str = 'unet'
    model_body_registry: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'unet': _UNet(),
            'unetpp': _UNetPP(),
            'unetppp': _UNetPPP(),
        }
    )
    bottleneck: str = 'conv'
    bottleneck_registry: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'conv': _UNetBottleneckConfig(),
            'transformer': _TransformerBottleneckConfig(),
            'hybrid': _HybridBottleneckConfig(),
        }
    )
    conditioners: list[str] = field(default_factory=lambda: [])
    conditioner_registry: dict[str, typing.Any] = field(
        default_factory=lambda: {
            'concat': _Concat(),
            'film': _FiLM()
        }
    )
    numeric_safety: _NumericSafety = field(default_factory=_NumericSafety)

    @property
    def unet_backbone_config(self) -> _UNetBackboneConfig:
        return _UNetBackboneConfig(
            body=self.model_body_registry[self.model_body],
            bottleneck=self.bottleneck_registry[self.bottleneck]
        )

    @property
    def conditioning_config(self) -> dict[str, _DomainTargetConfig]:
        assert all(c in self.conditioner_registry for c in self.conditioners)
        return {c: self.conditioner_registry[c] for c in self.conditioners}

    def set_base_channel(self, base_channel: int) -> None:
        '''Set `base channel` to the current model body.'''
        self.model_body_registry[self.model_body].base_ch = base_channel

    def validate(self) -> None:
        # model body selection
        if not self.model_body in self.model_body_registry:
            raise ValueError(
                f'Invalid model body: {self.model_body} ',
                f'expected: {list(self.model_body_registry.keys())}'
            )
        # conditioners
        if not all(c in self.conditioner_registry for c in self.conditioners):
            raise ValueError(
                f'Invalid conditioner(s): {self,self.conditioners} '
                f'expected: {list(self.conditioner_registry.keys())}'
            )
        # cross-check clamp range ordering
        lo, hi = self.numeric_safety.clamp_range
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError('Invalid clamp_range ordering or non-positive')
