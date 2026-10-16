# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Shared config fixtures for testing landseg.models.'''

# standard imports
import typing
# third-party imports
import pytest
# local imports
import landseg.configs.schema.models.backbones as backbones_schema
import landseg.configs.schema.models.conditioners as conditioners_schema


# ----- fixtures
@pytest.fixture
def mock_domain_config_factory():
    '''Return a domain target configuration building function.'''
    def _create(
        name: str = 'test_target',
        use_ids: bool = False,
        use_vec: bool = False,
        ids_embd_dims: int = 8,
        vec_proj_dims: int = 8,
        vec_proj_config: conditioners_schema.ProjectionConfig | None = None,
        conditioner_config: conditioners_schema.ConditionerAdapter | None = None
    ) -> conditioners_schema.DomainTargetConfig:
        if (
            name == 'film' or
            (conditioner_config and 'hidden_dim' in conditioner_config)
        ):
            return conditioners_schema.FiLM(
                name=name,
                use_ids=use_ids,
                use_vec=use_vec,
                ids_embd_dims=ids_embd_dims if use_ids else 0,
                vec_proj_dims=vec_proj_dims if use_vec else 0,
                vec_proj_config=vec_proj_config or {'use_mlp': False},
                conditioner_config=conditioner_config or {'hidden_dim': 128},
            )
        return conditioners_schema.Concat(
            name=name,
            use_ids=use_ids,
            use_vec=use_vec,
            ids_embd_dims=ids_embd_dims if use_ids else 0,
            vec_proj_dims=vec_proj_dims if use_vec else 0,
            vec_proj_config=vec_proj_config or {'use_mlp': False},
            conditioner_config=conditioner_config or {},
        )
    return _create


@pytest.fixture
def mock_backbone_config_factory():
    '''
    Return a mock backbone configuration building function.

    Defaults:
        `body_type='unet'` (`'unet'` | `'unetpp'` | `'unetppp'`),
        `base_ch=16`,
        `bottleneck='conv'` (`'conv'` | `'hybrid'` | `'transformer'`),
    '''
    def _create(
        body_type: str = 'unet',
        base_ch: int = 16,
        bottleneck: typing.Literal['conv', 'hybrid', 'transformer'] = 'conv',
    ) -> backbones_schema.UNetBackboneConfig:
        conv_params = backbones_schema.ConvParams(
            norm=None, gn_groups=None, p_drop=0.0
        )
        transformer_params = backbones_schema.Transformer(
            num_heads=1,
            mlp_ratio=2.0,
            dropout=0.0,
            attn_dropout=0.0,
        )

        body_cfg = backbones_schema.UNetBodyConfig(
            body=body_type,
            base_ch=base_ch,
            encoder_conv_params=conv_params,
            nodes_conv_params=conv_params,
            decoder_conv_params=conv_params,
        )
        match bottleneck:
            case 'conv':
                n_conv, n_transformer = 1, None
            case 'hybrid':
                n_conv, n_transformer = 1, 1
            case 'transformer':
                n_conv, n_transformer = None, 1
        btnk_cfg = backbones_schema.BottleneckConfig(
            variant=bottleneck,
            num_conv_blocks=n_conv,
            conv_params=conv_params,
            num_transformer_blocks=n_transformer,
            transformer_params=transformer_params,
        )
        return backbones_schema.UNetBackboneConfig(
            body=body_cfg, bottleneck=btnk_cfg
        )
    return _create
