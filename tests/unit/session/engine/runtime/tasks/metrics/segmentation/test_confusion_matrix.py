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

'''Unit tests for confusion matrix module (confusion_matrix.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.core as core
import landseg.session.engine.runtime.tasks.metrics.segmentation.confusion_matrix as cm_module


def test_confusion_matrix_init():
    '''
    Given: Number of classes, ignore index, parent class, and exclude classes.
    When: Instantiating `ConfusionMatrix`.
    Then: Correctly set attributes and initialize a zero matrix of shape (C, C).
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=3,
        ignore_index=255,
        parent_class_1b=1,
        exclude_class_1b=(3,)
    )

    assert cm.n_cls == 3
    assert cm.ignore_index == 255
    assert cm.parent_class_1b == 1
    assert cm.exclude_class_1b == (3,)
    assert cm.cm.shape == (3, 3)
    assert cm.cm.dtype == torch.int64
    assert torch.all(cm.cm == 0)


def test_confusion_matrix_update_basic():
    '''
    Given: Predictions of shape [B, C, H, W] and targets of shape [B, H, W].
    When: Calling `update`.
    Then: Increment confusion matrix entries according to (true, pred) pairs.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )

    # logits: B=1, C=2, H=1, W=2
    # pixel 0: argmax = class 0
    # pixel 1: argmax = class 1
    preds = torch.tensor([[[[10.0, -10.0]], [[-10.0, 10.0]]]], dtype=torch.float32)
    # targets (1-based): pixel 0 is target 1 (class 0), pixel 1 is target 2 (class 1)
    targets = torch.tensor([[[1, 2]]], dtype=torch.long)

    cm.update(preds, targets)

    # both pixels are correct matches
    # cm[0, 0] = 1, cm[1, 1] = 1
    assert cm.cm[0, 0].item() == 1
    assert cm.cm[1, 1].item() == 1
    assert cm.cm[0, 1].item() == 0
    assert cm.cm[1, 0].item() == 0


def test_confusion_matrix_update_ignore_index():
    '''
    Given: Targets containing ignore_index values.
    When: Calling `update`.
    Then: Skip ignored pixels from updating confusion matrix.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )

    preds = torch.tensor([[[[10.0, 10.0]], [[-10.0, -10.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[1, 255]]], dtype=torch.long)

    cm.update(preds, targets)

    assert cm.cm.sum().item() == 1
    assert cm.cm[0, 0].item() == 1


def test_confusion_matrix_update_parent_gating():
    '''
    Given: Optional `parent_raw_1b` kwarg and configured `parent_class_1b`.
    When: Calling `update`.
    Then: Only count pixels matching the required parent class.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=2,
        exclude_class_1b=None
    )

    preds = torch.tensor([[[[10.0, 10.0]], [[-10.0, -10.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[1, 1]]], dtype=torch.long)
    # parent raw labels: pixel 0 has parent 1, pixel 1 has parent 2
    parent_raw = torch.tensor([[[1, 2]]], dtype=torch.long)

    cm.update(preds, targets, parent_raw_1b=parent_raw)

    # only pixel 1 matches parent_class_1b=2
    assert cm.cm.sum().item() == 1
    assert cm.cm[0, 0].item() == 1


def test_confusion_matrix_update_all_ignored():
    '''
    Given: Batch where all targets are ignored.
    When: Calling `update`.
    Then: Return early without modifying confusion matrix.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )

    preds = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    targets = torch.full((1, 2, 2), 255, dtype=torch.long)

    cm.update(preds, targets)

    assert cm.cm.sum().item() == 0


def test_confusion_matrix_compute():
    '''
    Given: A populated `ConfusionMatrix`.
    When: Calling `compute`.
    Then: Return a locked `AccumulatedMetrics` object with IoUs.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )

    preds = torch.tensor([[[[10.0, -10.0]], [[-10.0, 10.0]]]], dtype=torch.float32)
    targets = torch.tensor([[[1, 2]]], dtype=torch.long)
    cm.update(preds, targets)

    metrics = cm.compute()

    assert isinstance(metrics, core.AccumulatedMetrics)
    assert metrics._locked
    assert metrics.ious['1'] == 1.0
    assert metrics.ious['2'] == 1.0
    assert metrics.mean == 1.0


def test_confusion_matrix_compute_exclude_classes():
    '''
    Given: `ConfusionMatrix` configured with `exclude_class_1b`.
    When: Calling `compute`.
    Then: Calculate active class IoUs (`ac_ious`) excluding specified classes.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=3,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=(3,)
    )

    # set up perfect matches for class 1 and 2
    cm.cm[0, 0] = 5
    cm.cm[1, 1] = 5
    cm.cm[2, 2] = 5

    metrics = cm.compute()

    assert '1' in metrics.ac_ious
    assert '2' in metrics.ac_ious
    assert '3' not in metrics.ac_ious
    assert metrics.ac_mean == 1.0


def test_confusion_matrix_compute_exclude_classes_out_of_range():
    '''
    Given: `ConfusionMatrix` configured with out-of-range `exclude_class_1b`.
    When: Calling `compute`.
    Then: Raise `IndexError`.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=(5,)
    )

    with pytest.raises(IndexError, match='Exclude classes out of index range'):
        cm.compute()


def test_confusion_matrix_compute_invalid_shape():
    '''
    Given: `ConfusionMatrix` with a non-square confusion matrix tensor.
    When: Calling `compute`.
    Then: Raise `ValueError`.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )
    cm.cm = torch.zeros((2, 3), dtype=torch.int64)

    with pytest.raises(ValueError, match='Confusion matrix must be a square 2D tensor'):
        cm.compute()


def test_confusion_matrix_reset():
    '''
    Given: A populated `ConfusionMatrix`.
    When: Calling `reset`.
    Then: Zero out confusion matrix tensor.
    '''
    cm = cm_module.ConfusionMatrix(
        num_classes=2,
        ignore_index=255,
        parent_class_1b=None,
        exclude_class_1b=None
    )

    cm.cm[0, 0] = 10
    cm.reset('cpu')

    assert cm.cm.sum().item() == 0
