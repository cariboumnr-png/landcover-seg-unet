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

'''Unit tests for Dice loss module (dice.py).'''

# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives.dice as dice_loss


def test_dice_loss_forward():
    '''
    Given: Logits and targets representing a segmentation task.
    When: `forward` is called on `DiceLoss`.
    Then: Return a valid scalar Dice loss tensor.
    '''
    loss_module = dice_loss.DiceLoss(smooth=1.0, ignore_index=255)
    # make logits favor class1 at top-left and class0 at others (perfect match)
    logits = torch.tensor(
        [[
            [[-10.0,  10.0], [ 10.0, 10.0]],
            [[ 10.0, -10.0], [-10.0, -10.0]]
        ]],
        dtype=torch.float32
    ) # [1, 2, 2, 2]
    # class 1 for top-left, class 0 for others (perfect match)
    targets = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.long)

    loss = loss_module(logits, targets, masks=None)

    # verify it is a scalar tensor and close to 0.0 since it is a perfect match
    assert loss.ndim == 0
    assert loss.item() < 0.1


def test_dice_loss_early_exit():
    '''
    Given: Targets containing only ignore_index.
    When: `forward` is called on `DiceLoss`.
    Then: Return a scalar zero tensor.
    '''
    loss_module = dice_loss.DiceLoss(smooth=1.0, ignore_index=255)
    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    targets = torch.full((1, 2, 2), 255, dtype=torch.long)

    loss = loss_module(logits, targets, masks=None)

    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_dice_loss_ignore_index():
    '''
    Given: Targets with some ignore_index pixels.
    When: `forward` is called on `DiceLoss`.
    Then: Exclude those pixels from the Dice calculation.
    '''
    loss_module = dice_loss.DiceLoss(smooth=1.0, ignore_index=255)
    # logits favor:
    # class 0 at (0,0), class 1 at (0,1), class 0 at (1,0), class 1 at (1,1)
    logits = torch.tensor(
        [[
            [[ 10.0, -10.0], [ 10.0, -10.0]],
            [[-10.0,  10.0], [-10.0,  10.0]]
        ]],
        dtype=torch.float32
    ) # [1, 2, 2, 2]

    # target matches logits except at (1,1) which is ignore_index
    targets = torch.tensor([[[0, 1], [0, 255]]], dtype=torch.long)
    loss_1 = loss_module(logits, targets, masks=None)

    # now target at (1,1) is class 0 (mismatch)
    targets_mismatch = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.long)
    loss_2 = loss_module(logits, targets_mismatch, masks=None)

    # loss_2 should be higher than loss_1
    # because (1,1) is a mismatch in targets_mismatch,
    # whereas in targets it was ignored.
    assert loss_1.item() < loss_2.item()


def test_dice_loss_with_masks():
    '''
    Given: A mask down-weighting mismatched pixels.
    When: `forward` is called on `DiceLoss`.
    Then: Return a lower loss due to the down-weighting.
    '''
    loss_module = dice_loss.DiceLoss(smooth=1.0, ignore_index=255)
    logits = torch.tensor(
        [[
            [[ 10.0,  10.0], [ 10.0,  10.0]],
            [[-10.0, -10.0], [-10.0, -10.0]]
        ]],
        dtype=torch.float32
    )
    # complete mismatch:
    # logits predict class 0 everywhere <-> targets is class 1 everywhere
    targets = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.long)

    # loss without mask
    loss_unmasked = loss_module(logits, targets, masks=None)

    # mask down-weights all pixels to 0.0
    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    loss_masked = loss_module(logits, targets, masks={0.0: mask})

    # masked loss should be 0.0 because of early exit due to zero weights sum
    assert loss_masked.item() == 0.0
    assert loss_unmasked.item() > 0.5
