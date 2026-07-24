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

'''Unit tests for composite loss module (composite.py).'''

# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.composite as composite


def test_composite_loss_init_no_losses(session_config):
    '''
    Given: A composite configuration with all loss weights equal to 0.0.
    When: Instantiating `CompositeLoss`.
    Then: Register no active losses and return 0.0 on forward.
    '''
    cfg = session_config.engine_tasks.loss_configs
    cfg.focal.weight = 0.0
    cfg.dice.weight = 0.0
    cfg.spectral.weight = 0.0
    cfg.tv.weight = 0.0

    loss_module = composite.CompositeLoss(cfg, ignore_index=255)

    assert len(loss_module.losses) == 0
    assert len(loss_module.weights) == 0

    p = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    t = torch.zeros((1, 3, 3), dtype=torch.long)
    out = loss_module(p, t)

    assert out.ndim == 0
    assert out.item() == 0.0


def test_composite_loss_init_all_losses(session_config):
    '''
    Given: A composite configuration enabling focal/dice/spectral/tv
        losses.
    When: Instantiating `CompositeLoss`.
    Then: Register all 4 primitive losses and compute weighted composite
        loss.
    '''
    cfg = session_config.engine_tasks.loss_configs
    cfg.focal.weight = 0.5
    cfg.dice.weight = 0.3
    cfg.spectral.weight = 0.1
    cfg.tv.weight = 0.1

    loss_module = composite.CompositeLoss(
        cfg,
        ignore_index=255,
        focal_alpha=[0.5, 0.5],
        spectral_band_indices=[0, 1]
    )

    assert len(loss_module.losses) == 4
    assert loss_module.weights == [0.5, 0.3, 0.1, 0.1]

    p = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    t = torch.zeros((1, 3, 3), dtype=torch.long)
    features = torch.randn((1, 4, 3, 3), dtype=torch.float32)

    out = loss_module(p, t, features=features)

    assert out.ndim == 0
    assert not torch.isnan(out)


def test_composite_loss_forward_weighted_sum(session_config):
    '''
    Given: Composite loss with focal and dice enabled.
    When: Calling `forward`.
    Then: Total loss == weight_focal*loss_focal+weight_dice*loss_dice.
    '''
    cfg = session_config.engine_tasks.loss_configs
    cfg.focal.weight = 0.6
    cfg.dice.weight = 0.4
    cfg.spectral.weight = 0.0
    cfg.tv.weight = 0.0

    loss_module = composite.CompositeLoss(cfg, ignore_index=255)

    p = torch.tensor(
        [[
            [[-10.0,  10.0], [ 10.0,  10.0]],
            [[ 10.0, -10.0], [-10.0, -10.0]]
        ]],
        dtype=torch.float32
    )
    t = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.long)

    total_loss = loss_module(p, t)

    # compute component losses individually
    loss_focal = loss_module.losses[0](p, t, masks=None, features=None)
    loss_dice = loss_module.losses[1](p, t, masks=None, features=None)

    expected = 0.6 * loss_focal + 0.4 * loss_dice
    assert torch.allclose(total_loss, expected)


def test_composite_loss_passes_masks(session_config):
    '''
    Given: Optional masks dictionary passed to `CompositeLoss.forward`.
    When: Calling `forward`.
    Then: Forward masks to component loss modules.
    '''
    cfg = session_config.engine_tasks.loss_configs
    cfg.focal.weight = 0.0
    cfg.dice.weight = 1.0
    cfg.spectral.weight = 0.0
    cfg.tv.weight = 0.0

    loss_module = composite.CompositeLoss(cfg, ignore_index=255)

    p = torch.tensor(
        [[
            [[ 10.0,  10.0], [ 10.0,  10.0]],
            [[-10.0, -10.0], [-10.0, -10.0]]
        ]],
        dtype=torch.float32
    )
    t = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.long)
    mask = torch.ones((1, 2, 2), dtype=torch.bool)

    # zero mask down-weights all pixels
    out = loss_module(p, t, masks={0.0: mask})

    assert out.item() == 0.0
