# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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

'''Unit tests for objective module (objective.py).'''

# standard imports
import dataclasses
# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.executor.objective as obj_mod


# ----- `_shift_1_to_0` helper tests
def test_shift_1_to_0_basic():
    '''
    Given: 1-based target tensor and ignore_index.
    When: Calling `_shift_1_to_0`.
    Then: Labels are shifted 1..K -> 0..K-1, keeping ignore_index unchanged.
    '''
    target = torch.tensor([[1, 2], [3, 0]])
    ignore_idx = 0

    shifted = obj_mod._shift_1_to_0(target, ignore_idx)

    expected = torch.tensor([[0, 1], [2, 0]])
    assert torch.equal(shifted, expected)


# ----- `_get_masks` helper tests
def test_get_masks_none():
    '''
    Given: Target tensor with no exclusion classes or parent gating.
    When: Calling `_get_masks`.
    Then: Return None.
    '''
    raw = torch.tensor([[1, 2]])

    masks = obj_mod._get_masks(raw=raw)

    assert masks is None


def test_get_masks_exclusion_and_parent():
    '''
    Given: Target tensor, exclusion classes, and parent gating details.
    When: Calling `_get_masks`.
    Then: Return mask dict mapping 0.05 for exclusion and 0.0 for parent gating.
    '''
    raw = torch.tensor([[1, 3], [2, 1]])
    parent_tensor = torch.tensor([[1, 1], [2, 1]])

    masks = obj_mod._get_masks(
        raw=raw,
        masked_cls=(3,),
        parent_tensor=parent_tensor,
        parent_cls_1b=1
    )

    assert masks is not None
    assert 0.05 in masks  # pylint: disable=unsupported-membership-test
    assert 0.0 in masks  # pylint: disable=unsupported-membership-test

    expected_excl = torch.tensor([[False, True], [False, False]])
    expected_parent = torch.tensor([[False, False], [True, False]])

    assert torch.equal(masks[0.05], expected_excl)  # pylint: disable=unsubscriptable-object
    assert torch.equal(masks[0.0], expected_parent)  # pylint: disable=unsubscriptable-object


# ----- `_prep_loss_compute` helper tests
def test_prep_loss_compute(mock_hlosses, mock_hspecs):
    '''
    Given: Head target tensor, HeadSpec, and `dummy_head_loss`.
    When: Calling `_prep_loss_compute`.
    Then: Return 0-based shifted target tensor and computed masks.
    '''
    head_target = torch.tensor([[1, 2]])
    head_spec = dataclasses.replace(mock_hspecs['head_1'], exclude_cls=(2,))

    target_0b, masks = obj_mod._prep_loss_compute(
        head_target=head_target,
        head_spec=head_spec,
        head_loss=mock_hlosses['head_1'],
        parent_tensor=None
    )

    assert torch.equal(target_0b, torch.tensor([[0, 1]]))
    assert masks is not None
    assert 0.05 in masks  # pylint: disable=unsupported-membership-test


# ----- `multihead_objective` public function tests
def test_multihead_objective_basic(mock_hlosses, mock_hspecs):
    '''
    Given: Multihead predictions, targets, features, and TrainingObjectives.
    When: Calling `multihead_objective`.
    Then: Calculate per-head loss, total loss, and return `_ObjectiveResults`.
    '''
    preds = {'head_1': torch.tensor([[[[1.0]], [[2.0]]]])} # [1, 2, 1, 1]
    targets = {'head_1': torch.tensor([[[1]]])} # [1, 1, 1]
    features = torch.tensor([[[[0.1]], [[0.2]]]]) # [1, 2, 1, 1]

    objectives =_get_objective(mock_hspecs, mock_hlosses)

    result = obj_mod.multihead_objective(
        multihead_preds=preds,
        multihead_targets=targets,
        features=features,
        objectives=objectives
    )

    assert isinstance(result, obj_mod._ObjectiveResults)
    assert 'head_1' in result.per_head_loss
    assert torch.isfinite(result.total)


def test_multihead_objective_with_regularizer(mock_hlosses, mock_hspecs):
    '''
    Given: `TrainingObjectives` configured with a regularizer.
    When: Calling `multihead_objective`.
    Then: Add regularizer output to total loss and store in regularization dict.
    '''
    preds = {'head_1': torch.tensor([[[[1.0]], [[2.0]]]])}
    targets = {'head_1': torch.tensor([[[1]]])}
    features = torch.tensor([[[[0.1]], [[0.2]]]])

    objectives = _get_objective(mock_hspecs, mock_hlosses, has_regularizer=True)

    result = obj_mod.multihead_objective(
        multihead_preds=preds,
        multihead_targets=targets,
        features=features,
        objectives=objectives
    )

    assert result.regularization['mtl_regularization'] == 0.5 # constant


def test_multihead_objective_nan_loss_raises(mock_hlosses, mock_hspecs):
    '''
    Given: Predictions resulting in NaN total loss.
    When: Calling `multihead_objective`.
    Then: Raise `RuntimeError` matching 'Contains NaN/Inf loss.'.
    '''
    preds = {'head_1': torch.tensor([[[[1.0]], [[float('nan')]]]])}
    targets = {'head_1': torch.tensor([[[1]]])}
    features = torch.tensor([[[[0.1]], [[0.2]]]])

    objectives = _get_objective(mock_hspecs, mock_hlosses)

    with pytest.raises(RuntimeError, match='Contains NaN/Inf loss'):
        obj_mod.multihead_objective(
            multihead_preds=preds,
            multihead_targets=targets,
            features=features,
            objectives=objectives
        )


# ----- internal helpers
def _get_objective(mock_hspecs, mock_hlosses, has_regularizer = False):

    class _MockRegularizer(torch.nn.Module):
        def __init__(self, reduction: str = 'mean', val: float = 0.5):
            super().__init__()
            self.reduction = reduction
            self.val = val

        def forward(
            self,
            multihead_preds: dict[str, torch.Tensor],
            multihead_targets: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            '''Mock forward.'''
            _ = multihead_preds, multihead_targets
            return torch.tensor(self.val)

    if not has_regularizer:
        return obj_mod.TrainingObjectives(
            headspecs={'head_1': mock_hspecs['head_1']},
            headlosses={'head_1': mock_hlosses['head_1']},
            mtl_regularization=None
        )
    return obj_mod.TrainingObjectives(
        headspecs={'head_1': mock_hspecs['head_1']},
        headlosses={'head_1': mock_hlosses['head_1']},
        mtl_regularization=_MockRegularizer() # type: ignore | mock
    )
