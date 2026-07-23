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

'''Unit tests for base primitive loss module (base.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.tasks.loss.primitives.base as base


def test_compose_pixel_weights_default():
    '''
    Given: Empty masks dictionary and no ignore_index.
    When: `_compose_pixel_weights` is called.
    Then: Return a tensor of all 1s matching the targets shape.
    '''
    targets = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    ws = base.PrimitiveLoss._compose_pixel_weights(
        masks=None,
        targets=targets,
        ignore_index=None,
        device=targets.device,
        dtype=torch.float32
    )

    # verify all weights are 1.0
    assert ws.shape == targets.shape
    assert torch.allclose(ws, torch.ones_like(ws))


def test_compose_pixel_weights_ignore_index():
    '''
    Given: An ignore_index value and target labels containing it.
    When: `_compose_pixel_weights` is called.
    Then: Return a weight map where targets matching `ignore_index` have
        weight 0.0.
    '''
    targets = torch.tensor([[1, 255], [3, 4]], dtype=torch.long)
    ws = base.PrimitiveLoss._compose_pixel_weights(
        masks=None,
        targets=targets,
        ignore_index=255,
        device=targets.device,
        dtype=torch.float32
    )

    # verify ignore_index has weight 0.0, others have 1.0
    expected = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(ws, expected)


def test_compose_pixel_weights_with_masks():
    '''
    Given: Down-weighting masks.
    When: `_compose_pixel_weights` is called.
    Then: Correctly apply down-weighting weights (min logic, clamping).
    '''
    targets = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    # mask 1 applies 0.5 weight to first row
    mask_1 = torch.tensor([[True, True], [False, False]], dtype=torch.bool)
    # mask 2 applies 0.2 weight to right column
    mask_2 = torch.tensor([[False, True], [False, True]], dtype=torch.bool)

    masks = {
        0.5: mask_1,
        0.2: mask_2,
        1.5: torch.ones_like(targets, dtype=torch.bool)  # clamped to 1.0
    }

    ws = base.PrimitiveLoss._compose_pixel_weights(
        masks=masks,
        targets=targets,
        ignore_index=None,
        device=targets.device,
        dtype=torch.float32
    )

    # pixel (0,0): weight 0.5 from mask_1
    # pixel (0,1): weight min(0.5, 0.2) = 0.2
    # pixel (1,0): weight 1.0 (no mask matches)
    # pixel (1,1): weight 0.2 from mask_2
    expected = torch.tensor([[0.5, 0.2], [1.0, 0.2]], dtype=torch.float32)
    assert torch.allclose(ws, expected)


def test_compose_pixel_weights_assertions():
    '''
    Given: Invalid mask key or invalid mask tensor.
    When: `_compose_pixel_weights` is called.
    Then: Raise `AssertionError`.
    '''
    targets = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    # invalid key - not int/float
    invalid_mask = {'not_a_number': torch.ones_like(targets, dtype=torch.bool)}
    with pytest.raises(AssertionError, match='Invalid mask keys'):
        _ = base.PrimitiveLoss._compose_pixel_weights(
            masks=invalid_mask, # type: ignore
            targets=targets,
            ignore_index=None,
            device=targets.device,
            dtype=torch.float32
        )

    # invalid value type - not a tensor
    invalid_mask = {0.5: [1, 2, 3]}
    with pytest.raises(AssertionError, match='Invalid mask type'):
        _ = base.PrimitiveLoss._compose_pixel_weights(
            masks=invalid_mask, # type: ignore
            targets=targets,
            ignore_index=None,
            device=targets.device,
            dtype=torch.float32
        )

    # mismatched shape - should be [2, 2]
    invalid_mask = {0.5: torch.ones((3, 3), dtype=torch.bool)}
    with pytest.raises(AssertionError, match='!= '):
        _ = base.PrimitiveLoss._compose_pixel_weights(
            masks=invalid_mask,
            targets=targets,
            ignore_index=None,
            device=targets.device,
            dtype=torch.float32
        )


def test_inputs_validation():
    '''
    Given: Inputs with invalid shapes.
    When: `_validate_inputs()` is called.
    Then: Raise `ValueError`.
    '''
    logits = torch.randn((1, 2, 3, 3), dtype=torch.float32)
    targets = torch.zeros((1, 3, 3), dtype=torch.long)

    # logits and targets validation (no features)
    with pytest.raises(ValueError, match=r'Expected \[B,C,H,W\] logits'):
        base.PrimitiveLoss._validate_inputs(
            torch.randn((1, 2, 3)),
            targets,
            None
        )

    with pytest.raises(ValueError, match=r'Expected \[B,H,W\] targets'):
        base.PrimitiveLoss._validate_inputs(
            logits,
            torch.zeros((1, 3)),
            None
        )

    with pytest.raises(ValueError, match=r'Batch.*logits & targets'):
        base.PrimitiveLoss._validate_inputs(
            logits,
            torch.zeros((2, 3, 3), dtype=torch.long),
            None
        )

    with pytest.raises(ValueError, match=r'Spatial.*logits & targets'):
        base.PrimitiveLoss._validate_inputs(
            logits,
            torch.zeros((1, 4, 4), dtype=torch.long),
            None
        )

    # when features are provided
    with pytest.raises(ValueError, match=r'Expected \[B,D,H,W\] features'):
        base.PrimitiveLoss._validate_inputs(
            logits,
            targets,
            features=torch.randn((1, 4, 3))
        )

    with pytest.raises(ValueError, match=r'Batch.*features & logits'):
        base.PrimitiveLoss._validate_inputs(
            logits,
            targets,
            features=torch.randn((2, 4, 3, 3))
        )

    with pytest.raises(ValueError, match=r'Spatial.*features & logits'):
        base.PrimitiveLoss._validate_inputs(
            logits,
            targets,
            features=torch.randn((1, 4, 4, 4))
        )
