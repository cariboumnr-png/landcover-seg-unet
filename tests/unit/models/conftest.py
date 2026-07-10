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
# pylint: disable=missing-class-docstring

'''Shared config fixtures and mock structures for testing landseg.models.'''

# standard imports
import dataclasses
# third-party imports
import pytest
# local imports
import landseg.models.core.config as config
import landseg.models.backbones.unet.components as components
import landseg.models.backbones.unet.body as body


# aliases
field = dataclasses.field


@dataclasses.dataclass(frozen=True)
class MockDomainTargetConfig:
    name: str
    use_ids: bool = False
    use_vec: bool = False
    ids_embd_dims: int = 0
    vec_proj_dims: int = 0
    vec_proj_config: config.DomainProjectionConfig = field(default_factory=dict)
    conditioner_config: config.DomainConditionerAdapterConfig = field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MockConvolutionParameters:
    norm: str | None = None
    gn_groups: int | None = None
    p_drop: float = 0.0


@dataclasses.dataclass(frozen=True)
class MockTransformerParameters:
    num_heads: int = 1
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    attn_dropout: float = 0.0


@dataclasses.dataclass(frozen=True)
class MockBottleneckConfig:
    variant: str = "conv"
    num_conv_blocks: int | None = 1
    conv_params: components.ConvolutionParameters | None = None
    num_transformer_blocks: int | None = None
    transformer_params: components.TransformerParameters | None = None


@dataclasses.dataclass(frozen=True)
class MockUNetBodyConfig:
    body: str = "unet"
    base_ch: int = 16
    encoder_conv_params: components.ConvolutionParameters = field(
        default_factory=MockConvolutionParameters
    )
    nodes_conv_params: components.ConvolutionParameters | None = None
    decoder_conv_params: components.ConvolutionParameters | None = None


@dataclasses.dataclass(frozen=True)
class MockUNetBackboneConfig:
    body: body.UNetBodyConfig
    bottleneck: components.BottleneckConfig


@pytest.fixture
def mock_domain_config_factory():
    def _create(
        name: str = "test_target",
        use_ids: bool = False,
        use_vec: bool = False,
        ids_embd_dims: int = 8,
        vec_proj_dims: int = 8,
        vec_proj_config: config.DomainProjectionConfig | None = None,
        conditioner_config: config.DomainConditionerAdapterConfig | None = None,
    ) -> MockDomainTargetConfig:
        return MockDomainTargetConfig(
            name=name,
            use_ids=use_ids,
            use_vec=use_vec,
            ids_embd_dims=ids_embd_dims,
            vec_proj_dims=vec_proj_dims,
            vec_proj_config=vec_proj_config or {},
            conditioner_config=conditioner_config or {},
        )
    return _create


@pytest.fixture
def mock_backbone_config_factory():
    def _create(
        body_type: str = "unet",
        base_ch: int = 16,
        bottleneck_variant: str = "conv",
    ) -> MockUNetBackboneConfig:
        conv_params = MockConvolutionParameters(
            norm=None, gn_groups=None, p_drop=0.0
        )
        body_cfg = MockUNetBodyConfig(
            body=body_type,
            base_ch=base_ch,
            encoder_conv_params=conv_params,
            nodes_conv_params=conv_params,
            decoder_conv_params=conv_params,
        )
        btnk_cfg = MockBottleneckConfig(
            variant=bottleneck_variant,
            num_conv_blocks=1,
            conv_params=conv_params,
        )
        return MockUNetBackboneConfig(body=body_cfg, bottleneck=btnk_cfg)
    return _create
