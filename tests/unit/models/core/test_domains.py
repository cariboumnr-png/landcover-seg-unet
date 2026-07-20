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

# pylint: disable=protected-access

'''Unit tests for domain context routing (domains.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.models.core.domains as domains


# ----- domain projection helpers
def test_make_projection_linear():
    '''
    Given: Dimension parameters and config disabling mlp.
    When: Running _make_projection.
    Then: Return a basic linear layer.
    '''
    proj = domains._make_projection(
        in_dim=4,
        out_dim=8,
        projection_config={'use_mlp': False}
    )
    assert isinstance(proj, torch.nn.Linear)
    assert proj.in_features == 4
    assert proj.out_features == 8


def test_make_projection_mlp_structure():
    '''
    Given: MLP configuration dictionaries.
    When: Running _make_projection.
    Then: Construct a Sequential MLP module.
    '''
    proj = domains._make_projection(
        in_dim=4,
        out_dim=8,
        projection_config={
            'use_mlp': True,
            'hidden_dim': 16,
            'num_hidden_layers': 2,
            'dropout': 0.1,
            'activation': 'relu'
        }
    )
    assert isinstance(proj, torch.nn.Sequential)
    assert len(proj) == 7
    assert isinstance(proj[0], torch.nn.Linear)
    assert proj[0].in_features == 4
    assert proj[0].out_features == 16
    assert isinstance(proj[1], torch.nn.ReLU)
    assert isinstance(proj[2], torch.nn.Dropout)
    assert proj[2].p == 0.1

    assert isinstance(proj[3], torch.nn.Linear)
    assert proj[3].in_features == 16
    assert proj[3].out_features == 16
    assert isinstance(proj[4], torch.nn.ReLU)
    assert isinstance(proj[5], torch.nn.Dropout)

    assert isinstance(proj[6], torch.nn.Linear)
    assert proj[6].in_features == 16
    assert proj[6].out_features == 8


def test_make_projection_unsupported_activation():
    '''
    Given: An invalid activation function name.
    When: Running _make_projection.
    Then: Raise a ValueError.
    '''
    with pytest.raises(ValueError, match='Unsupported activation: invalid'):
        domains._make_projection(
            in_dim=4,
            out_dim=8,
            projection_config={
                'use_mlp': True,
                'activation': 'invalid'
            }
        )


# ----- `DomainContextRouter`
def test_router_initialization(mock_domain_config_factory):
    '''
    Given: Target config maps.
    When: Initializing a DomainContextRouter.
    Then: Correctly allocate ids embeddings and vector projections.
    '''
    target_a = mock_domain_config_factory(use_ids=True, ids_embd_dims=16)
    target_b = mock_domain_config_factory(use_vec=True, vec_proj_dims=32)

    router = domains.DomainContextRouter(
        domain_ids_num=10,
        domain_vec_dim=5,
        targets={'target_a': target_a, 'target_b': target_b}
    )

    assert 'target_a' in router.ids_embd
    assert 'target_b' not in router.ids_embd
    assert 'target_b' in router.vec_proj
    assert 'target_a' not in router.vec_proj

    assert isinstance(router.ids_embd['target_a'], torch.nn.Embedding)
    assert router.ids_embd['target_a'].num_embeddings == 10
    assert router.ids_embd['target_a'].embedding_dim == 16

    assert isinstance(router.vec_proj['target_b'], torch.nn.Linear)
    assert router.vec_proj['target_b'].in_features == 5
    assert router.vec_proj['target_b'].out_features == 32


def test_router_forward(mock_domain_config_factory):
    '''
    Given: Domain indexes and embedding vectors.
    When: Forwarding through router.
    Then: Output projected payloads maps.
    '''
    target_a = mock_domain_config_factory(use_ids=True, ids_embd_dims=16)
    target_b = mock_domain_config_factory(use_vec=True, vec_proj_dims=32)

    router = domains.DomainContextRouter(
        domain_ids_num=10,
        domain_vec_dim=5,
        targets={'target_a': target_a, 'target_b': target_b}
    )

    ids = torch.randint(0, 10, (2,))
    vec = torch.randn(2, 5)

    routed = router.forward(ids, vec)

    assert 'target_a' in routed
    assert 'target_b' in routed

    assert isinstance(routed['target_a'], domains.DomainTargetPayload)
    assert routed['target_a'].ids_embd
    assert routed['target_a'].ids_embd.shape == (2, 16)
    assert routed['target_a'].vec_proj is None

    assert isinstance(routed['target_b'], domains.DomainTargetPayload)
    assert routed['target_b'].vec_proj
    assert routed['target_b'].vec_proj.shape == (2, 32)
    assert routed['target_b'].ids_embd is None


def test_router_forward_none_inputs(mock_domain_config_factory):
    '''
    Given: None values for domain inputs.
    When: Forwarding through router.
    Then: Return payload mapping populated with None tensors.
    '''
    target_a = mock_domain_config_factory(use_ids=True, ids_embd_dims=16)
    target_b = mock_domain_config_factory(use_vec=True, vec_proj_dims=32)

    router = domains.DomainContextRouter(
        domain_ids_num=10,
        domain_vec_dim=5,
        targets={'target_a': target_a, 'target_b': target_b}
    )

    routed = router.forward(None, None)

    assert routed['target_a'].ids_embd is None
    assert routed['target_a'].vec_proj is None
    assert routed['target_b'].ids_embd is None
    assert routed['target_b'].vec_proj is None
