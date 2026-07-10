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
# pylint: disable=protected-access

'''Unit tests for FiLM domain conditioning adapter (film.py).'''

# third-party imports
import torch
import pytest
# local imports
import landseg.models.core.conditioner.film as film
import landseg.models.core.domains as domains


# ----- FilmConditioner tests
def test_film_conditioner_pass_through():
    cond = film.FilmConditioner(embed_dim=0, bottleneck_ch=16)
    x = torch.randn(2, 16, 8, 8)
    out = cond(x, None)

    assert torch.allclose(x, out)
    assert cond.film is None


def test_film_conditioner_none_payload():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    x = torch.randn(2, 16, 8, 8)
    out = cond(x, None)

    assert torch.allclose(x, out)


def test_film_conditioner_empty_payload():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    x = torch.randn(2, 16, 8, 8)
    payload = domains.DomainTargetPayload(ids_embd=None, vec_proj=None)
    out = cond(x, payload)

    assert torch.allclose(x, out)


def test_film_conditioner_initialization():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16, hidden_dim=4)

    assert cond.embed_dim == 8
    assert cond.film is not None
    # film is Sequential: Linear(8, 4) -> ReLU -> Linear(4, 32)
    assert len(cond.film) == 3
    assert isinstance(cond.film[0], torch.nn.Linear)
    assert isinstance(cond.film[2], torch.nn.Linear)
    assert cond.film[0].out_features == 4
    assert cond.film[2].out_features == 32 # 2 * bottleneck_ch


def test_film_conditioner_build_z_layer_norm():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    ids_embd = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)

    z = cond._build_z(payload)
    assert z.shape == (2, 8)
    # verify layer norm properties (mean close to 0, var close to 1)
    assert torch.allclose(z.mean(dim=-1), torch.zeros(2), atol=1e-5)
    # LayerNorm standard deviation check
    # unbiased estimation vs population variance
    assert torch.allclose(z.var(dim=-1, unbiased=False), torch.ones(2), atol=1e-5)


def test_film_conditioner_build_z_mismatched():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    ids_embd = torch.randn(2, 8)
    vec_proj = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=vec_proj)

    with pytest.raises(ValueError, match='Mismatched payload dimensions'):
        cond._build_z(payload)


def test_film_conditioner_build_z_mismatched_output_dim():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=16)
    ids_embd = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)
    
    with pytest.raises(ValueError, match='Expected embedding dim=8, got 6'):
        cond._build_z(payload)


def test_film_conditioner_forward_modulation():
    cond = film.FilmConditioner(embed_dim=8, bottleneck_ch=4)
    x = torch.ones(2, 4, 8, 8)

    # mock film network outputs
    # output dimension is 2 * bottleneck_ch = 8.
    # we want gamma=0.5 (first 4 channels) and beta=0.2 (next 4 channels)
    # so we mock the forward pass of self.film
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
    cond.film = MockFilmSeq()

    out = cond(x, payload)
    # out = x * (1.0 + gamma) + beta
    # out = 1.0 * (1.0 + 0.5) + 0.2 = 1.7
    assert out.shape == (2, 4, 8, 8)
    assert torch.allclose(out, torch.full((2, 4, 8, 8), 1.7))
