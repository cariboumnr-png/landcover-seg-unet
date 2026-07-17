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

'''Unit tests for FiLM domain conditioning adapter (film.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.models.core.conditioner.film as film
import landseg.models.core.domains as domains


# ----- `FilmConditioner` tests
def test_film_conditioner_pass_through():
    '''
    Given: FilmConditioner initialized with embed_dim=0.
    When: Running forward pass.
    Then: Return the input tensor unchanged.
    '''
    cond = film.FilmConditioner(embed_dim=0, bottleneck_ch=16)
    x = torch.randn(2, 16, 8, 8)
    out = cond(x, None)

    assert torch.allclose(x, out)
    assert cond.film is None


def test_film_conditioner_none_payload():
    '''
    Given: FilmConditioner with non-zero embed_dim but a None payload.
    When: Running forward pass.
    Then: Pass through the input tensor unmodified.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    x = torch.randn(2, 16, 8, 8)
    out = cond(x, None)

    assert torch.allclose(x, out)


def test_film_conditioner_empty_payload():
    '''
    Given: FilmConditioner with non-zero embed_dim but empty payload.
    When: Running forward pass.
    Then: Pass through the input tensor unmodified.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    x = torch.randn(2, 16, 8, 8)
    payload = domains.DomainTargetPayload(ids_embd=None, vec_proj=None)
    out = cond(x, payload)

    assert torch.allclose(x, out)


def test_film_conditioner_initialization():
    '''
    Given: Bottleneck channels and hidden dimensions.
    When: Initializing a FilmConditioner.
    Then: Construct the linear projection MLP modules correctly.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16, hidden_dim=4)

    assert cond.embed_dim == 8
    assert cond.film is not None
    assert len(cond.film) == 3
    assert isinstance(cond.film[0], torch.nn.Linear)
    assert isinstance(cond.film[2], torch.nn.Linear)
    assert cond.film[0].out_features == 4
    assert cond.film[2].out_features == 32


def test_film_conditioner_build_z_layer_norm():
    '''
    Given: ID embedding payload.
    When: Building normalized embedding context z.
    Then: Standardize outputs to 0 mean and unit variance.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    ids_embd = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)

    z = cond._build_z(payload)
    assert isinstance(z, torch.Tensor)
    assert z.shape == (2, 8)
    mean = z.mean(dim=-1)
    var = z.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_film_conditioner_build_z_mismatched():
    '''
    Given: A payload containing conflicting ids and projection inputs.
    When: Building the embedding context z.
    Then: Raise a ValueError indicating mismatched dimensions.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    ids_embd = torch.randn(2, 8)
    vec_proj = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=vec_proj)

    with pytest.raises(ValueError, match='Mismatched payload dimensions'):
        cond._build_z(payload)


def test_film_conditioner_build_z_mismatched_output_dim():
    '''
    Given: A payload with wrong embedding dimensionality.
    When: Building the embedding context z.
    Then: Raise a ValueError.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    ids_embd = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)

    with pytest.raises(ValueError, match='Expected embedding dim=8, got 6'):
        cond._build_z(payload)


def test_film_conditioner_forward_modulation():
    '''
    Given: Mocked linear scale and bias parameters from film network.
    When: Modulating inputs.
    Then: Compute affine transform scaling: x * (1.0 + gamma) + beta.
    '''
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=4)
    x = torch.ones(2, 4, 8, 8)

    ids_embd = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)

    class MockFilmSeq(torch.nn.Module):
        def forward(self, z):
            return torch.cat(
                [
                    torch.full((z.shape[0], 4), 0.5),
                    torch.full((z.shape[0], 4), 0.2)
                ], dim=1
            )
    cond.film = MockFilmSeq() # type: ignore

    out = cond(x, payload)
    assert out.shape == (2, 4, 8, 8)
    assert torch.allclose(out, torch.full((2, 4, 8, 8), 1.7))
