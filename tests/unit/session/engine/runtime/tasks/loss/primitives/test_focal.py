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

'''Unit tests for Focal loss module (focal.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives.focal as focal_loss


def test_focal_loss_forward():
    '''
    Given: Logits and targets with different reductions.
    When: `forward` is called on `FocalLoss`.
    Then: Return a valid loss tensor conforming to the reduction mode.
    '''
    logits = torch.tensor(
        [[
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 3.0], [2.0, 1.0]]
        ]],
        dtype=torch.float32
    )
    targets = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)

    # mean reduction
    loss_mean = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='mean',
        ignore_index=255
    )

    out_mean = loss_mean(logits, targets, masks=None)
    assert out_mean.ndim == 0

    # sum reduction
    loss_sum = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='sum',
        ignore_index=255
    )
    out_sum = loss_sum(logits, targets, masks=None)
    assert out_sum.ndim == 0
    assert out_sum.item() > out_mean.item()

    # none reduction
    loss_none = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='none',
        ignore_index=255
    )
    out_none = loss_none(logits, targets, masks=None)
    assert out_none.shape == (4,)  # flat number of valid pixels


def test_focal_loss_early_exit():
    '''
    Given: Targets containing only ignore_index.
    When: `forward` is called on `FocalLoss`.
    Then: Return a scalar zero tensor.
    '''
    loss_module = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='mean',
        ignore_index=255
    )
    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    targets = torch.full((1, 2, 2), 255, dtype=torch.long)

    loss = loss_module(logits, targets, masks=None)

    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_focal_loss_alpha_weighting():
    '''
    Given: Class weights alpha.
    When: `forward` is called on `FocalLoss`.
    Then: Scale the loss of different classes by the respective alpha weights.
    '''
    # configure alpha: class 0 has weight 0.1, class 1 has weight 0.9
    loss_weighted = focal_loss.FocalLoss(
        alpha=[0.1, 0.9],
        gamma=2.0,
        reduction='none',
        ignore_index=255
    )
    loss_unweighted = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='none',
        ignore_index=255
    )

    logits = torch.zeros((1, 2, 1, 2), dtype=torch.float32) # [B, C, H, W]
    targets = torch.tensor([[[0, 1]]], dtype=torch.long)

    out_weighted = loss_weighted(logits, targets, masks=None)
    out_unweighted = loss_unweighted(logits, targets, masks=None)

    # verify class 0 has 0.1 loss scale
    assert torch.allclose(out_weighted[0], out_unweighted[0] * 0.1)
    # verify class 1 has 0.9 loss scale
    assert torch.allclose(out_weighted[1], out_unweighted[1] * 0.9)


def test_focal_loss_gamma_effect():
    '''
    Given: Different gamma focus parameters.
    When: Logits represent high-confidence prediction versus
        low-confidence.
    Then: Ensure focal loss with gamma > 0 suppresses easy predictions
        compared to cross-entropy (gamma=0).
    '''
    # gamma = 0 (equivalent to standard Cross-Entropy)
    loss_ce = focal_loss.FocalLoss(
        alpha=None,
        gamma=0.0,
        reduction='none',
        ignore_index=255
    )
    # gamma = 2
    loss_focal = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='none',
        ignore_index=255
    )

    # pixel 0: correct and confident (p_t ~ 0.95);
    # pixel 1: incorrect/hard (p_t ~ 0.05)
    logits = torch.tensor([[[[3.0, -3.0]], [[-3.0, 3.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[0, 0]]], dtype=torch.long)  # both true label is 0

    out_ce = loss_ce(logits, targets, masks=None)
    out_focal = loss_focal(logits, targets, masks=None)

    # for confident pixel, focal loss should be significantly lower than CE loss
    ratio_confident = out_focal[0] / out_ce[0]
    # for hard pixel, focal loss suppression ratio should be much closer to 1
    ratio_hard = out_focal[1] / out_ce[1]

    assert ratio_confident < ratio_hard


def test_focal_loss_target_out_of_range():
    '''
    Given: Target index out of bounds of class count.
    When: `forward` is called on `FocalLoss`.
    Then: Raise `IndexError`.
    '''
    loss_module = focal_loss.FocalLoss(
        alpha=None,
        gamma=2.0,
        reduction='mean',
        ignore_index=255
    )
    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)  # C = 2 classes
    targets = torch.tensor([[[0, 2], [0, 0]]], dtype=torch.long)
    # target index 2 is invalid (max class should be < 2)

    with pytest.raises(IndexError, match='target out of range'):
        _ = loss_module(logits, targets, masks=None)
