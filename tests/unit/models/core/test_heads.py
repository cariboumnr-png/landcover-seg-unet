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

'''Unit tests for HeadManager (heads.py).'''

# third-party imports
import torch
# local imports
import landseg.models.core.heads as heads


# ----- `HeadManager` initialization
def test_head_manager_initialization():
    in_ch = 16
    heads_dict = {'head_a': 3, 'head_b': 5}
    hm = heads.HeadManager(in_ch, heads_dict)

    assert isinstance(hm.outc, torch.nn.ModuleDict)
    assert 'head_a' in hm.outc
    assert 'head_b' in hm.outc

    assert isinstance(hm.outc['head_a'], torch.nn.Conv2d)
    assert hm.outc['head_a'].in_channels == in_ch
    assert hm.outc['head_a'].out_channels == 3

    assert hm.outc['head_b'].in_channels == in_ch
    assert hm.outc['head_b'].out_channels == 5

    assert set(hm.active) == {'head_a', 'head_b'}
    assert hm.frozen is None


# ----- `HeadManager` forward passes
def test_head_manager_forward_active_heads():
    in_ch = 8
    heads_dict = {'head_a': 2, 'head_b': 3}
    hm = heads.HeadManager(in_ch, heads_dict)

    x = torch.randn(2, in_ch, 16, 16)
    out = hm.forward(
        x,
        active_heads=['head_a'],
        logit_adjust={},
        logit_adjust_alpha=1.0
    )

    assert 'head_a' in out
    assert 'head_b' not in out
    assert out['head_a'].shape == (2, 2, 16, 16)


def test_head_manager_forward_default_active_heads():
    in_ch = 8
    heads_dict = {'head_a': 2, 'head_b': 3}
    hm = heads.HeadManager(in_ch, heads_dict)

    x = torch.randn(2, in_ch, 16, 16)
    out = hm.forward(
        x,
        active_heads=None,
        logit_adjust={},
        logit_adjust_alpha=1.0
    )

    assert 'head_a' in out
    assert 'head_b' in out
    assert out['head_a'].shape == (2, 2, 16, 16)
    assert out['head_b'].shape == (2, 3, 16, 16)


# ----- `HeadManager` freezing
def test_head_manager_freeze():
    in_ch = 8
    heads_dict = {'head_a': 2, 'head_b': 3}
    hm = heads.HeadManager(in_ch, heads_dict)

    hm.freeze(['head_a'])

    for p in hm.outc['head_a'].parameters():
        assert not p.requires_grad

    for p in hm.outc['head_b'].parameters():
        assert p.requires_grad


# ----- `HeadManager` logit adjustment
def test_head_manager_logit_adjustment():
    logits = torch.ones(1, 2, 1, 1)
    prior = torch.tensor([0.5, -0.5]).view(1, 2, 1, 1)
    logit_adjust = {'head_a': prior}

    # standard adjustment
    adjusted = heads.HeadManager._apply_logit_adjust(
        'head_a', logits, logit_adjust=logit_adjust, la_alpha=1.0
    )
    expected = logits + prior
    assert torch.allclose(adjusted, expected)

    # test custom scaling factor alpha
    adjusted_scaled = heads.HeadManager._apply_logit_adjust(
        'head_a', logits, logit_adjust=logit_adjust, la_alpha=2.0
    )
    expected_scaled = logits + 2.0 * prior
    assert torch.allclose(adjusted_scaled, expected_scaled)

    # test alpha = 0.0 disables adjustment
    adjusted_zero = heads.HeadManager._apply_logit_adjust(
        'head_a', logits, logit_adjust=logit_adjust, la_alpha=0.0
    )
    assert torch.allclose(adjusted_zero, logits)

    # test missing prior does not adjust
    adjusted_missing = heads.HeadManager._apply_logit_adjust(
        'head_b', logits, logit_adjust=logit_adjust, la_alpha=1.0
    )
    assert torch.allclose(adjusted_missing, logits)


# ----- `HeadManager`numerical stability
def test_head_manager_nan_to_num():
    in_ch = 4
    hm = heads.HeadManager(in_ch, {'head_a': 1})

    # mock a convolutional output with non-finite values
    x = torch.zeros(1, in_ch, 4, 4)
    hm.outc['head_a'].weight.data.fill_(0.0)
    hm.outc['head_a'].bias.data.fill_(float('nan'))

    out = hm.forward(
        x,
        active_heads=['head_a'],
        logit_adjust={},
        logit_adjust_alpha=1.0
    )
    assert torch.all(out['head_a'] == 0.0)

    hm.outc['head_a'].bias.data.fill_(float('inf'))
    out_pos = hm.forward(
        x,
        active_heads=['head_a'],
        logit_adjust={},
        logit_adjust_alpha=1.0
    )
    assert torch.all(out_pos['head_a'] == 1e4)

    hm.outc['head_a'].bias.data.fill_(float('-inf'))
    out_neg = hm.forward(
        x,
        active_heads=['head_a'],
        logit_adjust={},
        logit_adjust_alpha=1.0
    )
    assert torch.all(out_neg['head_a'] == -1e4)
