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

'''Unit tests for spatial domain concatenation adapter (concat.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.models.core.conditioner.concat as concat
import landseg.models.core.domains as domains


# ----- `ConcatAdapter` tests
def test_concat_adapter_pass_through():
    '''
    Given: ConcatAdapter initialized with concat_dim=0.
    When: Running forward pass.
    Then: Return the input tensor unchanged.
    '''
    adapter = concat.ConcatAdapter(concat_dim=0)
    x = torch.randn(2, 4, 16, 16)
    out = adapter(x, None)

    assert torch.allclose(x, out)


def test_concat_adapter_missing_payload():
    '''
    Given: ConcatAdapter with non-zero concat_dim and a None payload.
    When: Running forward pass.
    Then: Raise a ValueError.
    '''
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)

    with pytest.raises(ValueError, match='Expected DomainTargetPayload'):
        adapter(x, None)


def test_concat_adapter_empty_payload():
    '''
    Given: ConcatAdapter with non-zero concat_dim and empty payload.
    When: Running forward pass.
    Then: Raise a ValueError.
    '''
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    payload = domains.DomainTargetPayload(ids_embd=None, vec_proj=None)

    with pytest.raises(ValueError, match='neither ids_embd nor vec_proj'):
        adapter(x, payload)


def test_concat_adapter_ids_only():
    '''
    Given: Payload containing only ID embeddings.
    When: Running forward pass.
    Then: Concatenate expanded ID embedding values along channel dim.
    '''
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)
    out = adapter(x, payload)

    assert out.shape == (2, 12, 16, 16)
    expected_slice = ids_embd.view(2, 8, 1, 1).expand(2, 8, 16, 16)
    assert torch.allclose(out[:, 4:], expected_slice)


def test_concat_adapter_vec_only():
    '''
    Given: Payload containing only vector projections.
    When: Running forward pass.
    Then: Concatenate expanded vector projections along channel dim.
    '''
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    vec_proj = torch.randn(2, 8)
    payload = domains.DomainTargetPayload(ids_embd=None, vec_proj=vec_proj)
    out = adapter(x, payload)

    assert out.shape == (2, 12, 16, 16)
    expected_slice = vec_proj.view(2, 8, 1, 1).expand(2, 8, 16, 16)
    assert torch.allclose(out[:, 4:], expected_slice)


def test_concat_adapter_both():
    '''
    Given: Payload containing both ID embedding and vector projections.
    When: Running forward pass.
    Then: Concatenate the element-wise sum of both payloads along channel
        dim.
    '''
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
    '''
    Given: Conflicting shapes for both ID embedding and vector projection.
    When: Running forward pass.
    Then: Raise a ValueError indicating mismatched dimensions.
    '''
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 8)
    vec_proj = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=vec_proj)

    with pytest.raises(ValueError, match='Mismatched payload dimensions'):
        adapter(x, payload)


def test_concat_adapter_mismatched_output_dim():
    '''
    Given: An ID embedding payload of wrong dimensionality.
    When: Running forward pass.
    Then: Raise a ValueError.
    '''
    adapter = concat.ConcatAdapter(concat_dim=8)
    x = torch.randn(2, 4, 16, 16)
    ids_embd = torch.randn(2, 6)
    payload = domains.DomainTargetPayload(ids_embd=ids_embd, vec_proj=None)

    with pytest.raises(ValueError, match='Expected domain dim=8, got 6'):
        adapter(x, payload)
