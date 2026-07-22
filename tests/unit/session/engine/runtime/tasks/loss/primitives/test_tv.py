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

'''Unit tests for Total Variation loss module (tv.py).'''

# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives.tv as tv_loss


def test_tv_loss_forward():
    '''
    Given: Logits and targets.
    When: `forward` is called on `TotalVariationLoss`.
    Then: Return a valid scalar Total Variation loss tensor.
    '''
    loss_module = tv_loss.TotalVariationLoss(ignore_index=None)
    # B=1, C=2, H=3, W=3
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)

    loss = loss_module(logits, targets, masks=None)

    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_tv_loss_early_exit():
    '''
    Given: Mappings / ignore indices that mask all pixel pairs.
    When: `forward` is called on `TotalVariationLoss`.
    Then: Return a scalar zero tensor.
    '''
    loss_module = tv_loss.TotalVariationLoss(ignore_index=255)
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.full((1, 3, 3), 255, dtype=torch.long)  # all pixels ignored

    loss = loss_module(logits, targets, masks=None)

    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_tv_loss_ignore_index():
    '''
    Given: Targets containing some ignore_index.
    When: `forward` is called on `TotalVariationLoss`.
    Then: Correctly mask those boundaries, changing the final TV loss.
    '''
    loss_module_no_ignore = tv_loss.TotalVariationLoss(ignore_index=None)
    loss_module_ignore = tv_loss.TotalVariationLoss(ignore_index=255)

    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    targets[0, 1, 1] = 255  # ignore center pixel

    loss_unmasked = loss_module_no_ignore(logits, targets, masks=None)
    loss_masked = loss_module_ignore(logits, targets, masks=None)

    assert loss_unmasked.item() != loss_masked.item()
