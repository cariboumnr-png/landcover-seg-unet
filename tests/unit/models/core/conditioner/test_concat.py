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
# pylint: disable=protected-access

'''Unit tests for spatial domain concatenation adapter (concat.py).'''

# third-party imports
import torch
import pytest
# local imports
import landseg.models.core.conditioner.concat as concat
import landseg.models.core.domains as domains


# ----- ConcatAdapter tests
def test_concat_adapter_pass_through():
    adapter = concat.ConcatAdapter(concat_dim=0)
    x = torch.randn(2, 4, 16, 16)
    out = adapter(x, None)

    assert torch.allclose(x, out)


def test_concat_adapter_missing_payload():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)

    with pytest.raises(ValueError, match='Expected DomainTargetPayload'):
        adapter(x, None)


def test_concat_adapter_empty_payload():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    payload = domains.DomainTargetPayload(ids_embd=None, vec_proj=None)

    with pytest.raises(ValueError, match='neither ids_embd nor vec_proj'):
        adapter(x, payload)


def test_concat_adapter_ids_only():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)
    out = adapter(x, payload)

    assert out.shape == (2, 12, 16, 16)
    # verify concatenated content
    expected_slice = ids_embd.view(2, 8, 1, 1).expand(2, 8, 16, 16)
    assert torch.allclose(out[:, 4:], expected_slice)


def test_concat_adapter_vec_only():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    vec_proj = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=None, vec_proj=vec_proj)
    out = adapter(x, payload)

    assert out.shape == (2, 12, 16, 16)
    expected_slice = vec_proj.view(2, 8, 1, 1).expand(2, 8, 16, 16)
    assert torch.allclose(out[:, 4:], expected_slice)


def test_concat_adapter_both():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 8)
    vec_proj = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=vec_proj)
    out = adapter(x, payload)

    assert out.shape == (2, 12, 16, 16)
    expected_sum = ids_embd + vec_proj
    expected_slice = expected_sum.view(2, 8, 1, 1).expand(2, 8, 16, 16)
    assert torch.allclose(out[:, 4:], expected_slice)


def test_concat_adapter_mismatched_both():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 8)
    vec_proj = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=vec_proj)

    with pytest.raises(ValueError, match='Mismatched payload dimensions'):
        adapter(x, payload)


def test_concat_adapter_mismatched_output_dim():
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)

    with pytest.raises(ValueError, match='Expected domain dim=8, got 6'):
        adapter(x, payload)
