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

'''Unit tests for regularization module (regularization.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.session.engine.runtime.tasks.regularization.consistency as consistency


def test_regularizer_init_invalid_reduction(session_config):
    '''
    Given: `ConsistencyRegularizer` init args with invalid reduction.
    When: Class initiation.
    Then: Raise `ValueError`.
    '''
    reg_cfg = session_config.engine_tasks.mtl_reg_configs # from default fixture
    reg_cfg.consistency_reduction = 'invalid_reduction_method' # manual

    with pytest.raises(ValueError, match='Invalid reduction'):
        _ = consistency.ConsistencyRegularizer(None, reg_cfg, ignore_index=0)


def test_regularizer_duplicated_constrains(session_config, mock_constraint):
    '''
    Given: `ConsistencyRegularizer` init args with duplicated constraints.
    When: Class initiation.
    Then: Raise `ValueError`.
    '''
    reg_cfg = session_config.engine_tasks.mtl_reg_configs
    cc = [mock_constraint(name='rule_1'), mock_constraint(name='rule_1')]
    # only detects name collision

    with pytest.raises(ValueError, match='Duplicated consistency constraints'):
        _ = consistency.ConsistencyRegularizer(cc, reg_cfg, ignore_index=0)


@pytest.mark.parametrize(
    ('reduction', 'expected'),
    [
        ('mean', torch.tensor(0.5)),
        ('sum', torch.tensor(1.0)),
        ('none', torch.tensor([0.5])),
    ]
)
def test_regularizer_forward_reductions(
    session_config,
    mock_constraint,
    reduction,
    expected
):
    '''
    Given: One active constraint with two valid pixels and uniform logits.
    When: `ConsistencyRegularizer.forward()` is called with each reduction.
    Then: Return the correctly reduced and lambda-scaled penalty.
    '''
    reg_cfg = session_config.engine_tasks.mtl_reg_configs
    reg_cfg.consistency_lambda = 2.0
    reg_cfg.consistency_reduction = reduction

    cons = mock_constraint(
        trigger_val=1,
        forbidden=[1]
    )
    regularizer = consistency.ConsistencyRegularizer(
        [cons],
        reg_cfg,
        ignore_index=255
    )

    # Each pixel has:
    # P(source=1) * P(target=1) = 0.5 * 0.5 = 0.25.
    # Two pixels produce:
    # mean = 0.25, sum = 0.50.
    # Applying lambda=2 gives mean=0.50 and sum=1.00.
    logits = {
        'head_1': torch.ones(1, 2, 1, 2),
        'head_2': torch.ones(1, 2, 1, 2),
    }
    targets = {
        'head_1': torch.zeros(1, 1, 2, dtype=torch.long),
        'head_2': torch.zeros(1, 1, 2, dtype=torch.long),
    }

    result = regularizer(logits, targets)

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ('reduction', 'expected_shape'),
    [
        ('mean', torch.Size([])),
        ('sum', torch.Size([])),
        ('none', torch.Size([0])),
    ]
)
def test_regularizer_forward_empty_logits(
    session_config,
    reduction,
    expected_shape
):
    '''
    Given: An empty logits dictionary.
    When: `ConsistencyRegularizer.forward()` is called.
    Then: Return an unscaled zero scalar or empty tensor for the reduction.
    '''
    reg_cfg = session_config.engine_tasks.mtl_reg_configs
    reg_cfg.consistency_lambda = 2.0
    reg_cfg.consistency_reduction = reduction

    regularizer = consistency.ConsistencyRegularizer(
        None,
        reg_cfg,
        ignore_index=255
    )

    result = regularizer({}, {})

    assert result.shape == expected_shape
    assert result.numel() == 0 or result.item() == 0.0


@pytest.mark.parametrize(
    ('reduction', 'expected_shape'),
    [
        ('mean', torch.Size([])),
        ('sum', torch.Size([])),
        ('none', torch.Size([0])),
    ]
)
def test_regularizer_forward_no_constraints(
    session_config,
    reduction,
    expected_shape
):
    '''
    Given: Valid logits but no configured constraints.
    When: `ConsistencyRegularizer.forward()` is called.
    Then: Return a device-aware zero scalar or empty tensor.
    '''
    reg_cfg = session_config.engine_tasks.mtl_reg_configs
    reg_cfg.consistency_reduction = reduction

    regularizer = consistency.ConsistencyRegularizer(
        None,
        reg_cfg,
        ignore_index=255
    )

    logits = {
        'head_1': torch.randn(1, 2, 2, 2, dtype=torch.float64),
    }

    result = regularizer(logits, {})

    assert result.shape == expected_shape
    assert result.dtype == logits['head_1'].dtype
    assert result.device == logits['head_1'].device
    assert result.numel() == 0 or result.item() == 0.0


def test_regularizer_by_constraint(session_config, mock_constraint):
    '''
    Given: Two active named constraints with known probabilities.
    When: `ConsistencyRegularizer.by_constraint()` is called.
    Then: Return unscaled mean penalties keyed by constraint name.
    '''
    reg_cfg = session_config.engine_tasks.mtl_reg_configs
    reg_cfg.consistency_lambda = 10.0

    constraints = [
        mock_constraint(
            name='rule_1',
            trigger_val=0,
            forbidden=[1]
        ),
        mock_constraint(
            name='rule_2',
            trigger_val=1,
            forbidden=[0]
        ),
    ]
    regularizer = consistency.ConsistencyRegularizer(
        constraints,
        reg_cfg,
        ignore_index=255
    )

    # Uniform two-class logits give 0.5 * 0.5 = 0.25 for both rules.
    logits = {
        'head_1': torch.ones(1, 2, 1, 1),
        'head_2': torch.ones(1, 2, 1, 1),
    }
    targets = {
        'head_1': torch.zeros(1, 1, 1, dtype=torch.long),
        'head_2': torch.zeros(1, 1, 1, dtype=torch.long),
    }

    result = regularizer.by_constraint(logits, targets)

    assert set(result) == {'rule_1', 'rule_2'}
    torch.testing.assert_close(result['rule_1'], torch.tensor(0.25))
    torch.testing.assert_close(result['rule_2'], torch.tensor(0.25))


@pytest.mark.parametrize(
    ('logits', 'labels', 'match'),
    [
       (
           torch.randn(2, 3, 8),
           torch.zeros(2, 8, 8),
           r'logits must be \[B,C,H,W\]'
       ),
       (
           torch.randn(2, 3, 8, 8),
           torch.zeros(8, 8),
           r'labels must be \[B,H,W\]'
       ),
       (
           torch.randn(2, 3, 8, 8),
           torch.zeros(3, 8, 8),
           'batch size mismatch'
       ),
       (
           torch.randn(2, 3, 8, 8),
           torch.zeros(2, 7, 7),
           r'H \* W mismatch'
       ),
    ]
)
def test_validate_head_pair(logits, labels, match):
    '''
    Given: Input logits and labels pair
    When: Private function _validate_head_pair() is called
    Then: Correctly raise when input logits/targets are invalid.
    '''
    with pytest.raises(ValueError, match=match):
        consistency._validate_head_pair('', logits, labels)


def test_validate_shapes(mock_constraint):
    '''
    Given: Contraint and input tensors (source and target pairs).
    When: Private function _validate_shapes() is called.
    Then: Correctly raise when inputs are invalid or incompatible with
        then constraint.
    '''
    # checks between input source and target shapes
    valid_cons = mock_constraint()

    invalid_tensors = {
        'source_logits': torch.randn(2, 3, 8, 8), # 2 batches
        'source_labels': torch.zeros(2, 8, 8, dtype=torch.long),
        'target_logits': torch.randn(4, 3, 8, 8), # 4 batches
        'target_labels': torch.zeros(4, 8, 8, dtype=torch.long),
    }
    with pytest.raises(ValueError, match='mismatched batches'):
        consistency._validate_shapes(constraint=valid_cons, **invalid_tensors)

    invalid_tensors = {
        'source_logits': torch.randn(2, 3, 8, 8), # 8*8
        'source_labels': torch.zeros(2, 8, 8, dtype=torch.long),
        'target_logits': torch.randn(2, 3, 16, 16), # 16*16
        'target_labels': torch.zeros(2, 16, 16, dtype=torch.long),
    }
    with pytest.raises(ValueError, match=r'mismatched H \* W'):
        consistency._validate_shapes(constraint=valid_cons, **invalid_tensors)

    # checks between constraint and channel number of input tensors
    valid_tensors = {
        'source_logits': torch.randn(2, 3, 8, 8),
        'source_labels': torch.zeros(2, 8, 8, dtype=torch.long),
        'target_logits': torch.randn(2, 3, 8, 8),
        'target_labels': torch.zeros(2, 8, 8, dtype=torch.long),
    }

    invalid_cons = mock_constraint(trigger_val=999) # invalid trigger 999
    with pytest.raises(IndexError, match='trigger class index'):
        consistency._validate_shapes(constraint=invalid_cons, **valid_tensors)

    invalid_cons = mock_constraint(forbidden=[1, 999]) # invalid forbidden 999
    with pytest.raises(IndexError, match='forbidden class indices'):
        consistency._validate_shapes(constraint=invalid_cons, **valid_tensors)


def test_invalid_stats_prob_single_forbidden_class():
    '''
    Given: Source and target logits for a single pixel with two classes.
    When: source=class_0 (trigger) * target=class_1 (forbidden)
    Then: The result is 0.25 (0.5 * 0.5).
    '''
    # 1 pixel with 2 classes for both source and target [1, 2, 1, 1]
    source_logits = torch.tensor([[
        [[1.0]],
        [[1.0]]
    ]])
    target_logits = torch.tensor([[
        [[1.0]],
        [[1.0]]
    ]])

    result = consistency._invalid_state_probability(
        trigger_val=0, # where class_0 is predicted in source
        forbidden=(1,), # at the same location target should not be class_1
        source_logits=source_logits,
        target_logits=target_logits
    )
    torch.testing.assert_close(result.item(), 0.25) # 0.5 * 0.5


def test_invalid_stats_prob_multiple_forbidden_classes():
    '''
    Given: Source and target logits for a single pixel with three classes.
    When: source=class_0 (trigger) * target=class_1 & class_2 (forbidden)
    Then: The result is 2/9  (1/3 * 2/3).
    '''
    # 1 pixel with 3 classes for both source and target [1, 3, 1, 1]
    source_logits = torch.tensor([[
        [[1.0]],
        [[1.0]],
        [[1.0]]
    ]])
    target_logits = torch.tensor([[
        [[1.0]],
        [[1.0]],
        [[1.0]]
    ]])

    result = consistency._invalid_state_probability(
        trigger_val=0, # where class_0 is predicted in source
        forbidden=(1, 2), # at the same location target should not be class_1/2
        source_logits=source_logits,
        target_logits=target_logits
    )
    torch.testing.assert_close(result.item(), 2 / 9) # 1/3 * 2/3


def test_constraint_value_early_exit_not_all_tensors_valid(mock_constraint):
    '''
    Given: Incomplete logits and target dictionaries.
    When: Private function _constraint_value() is called.
    Then: Returns `None`.
    '''
    cons = mock_constraint(source_head='head1', target_head='head2')
    complete_logits = {'head1': torch.tensor([0]), 'head2': torch.tensor([0])}
    incomplete_logits = {'head1': torch.tensor([0])} # missing 'head2'
    complete_target = {'head1': torch.tensor([0]), 'head2': torch.tensor([0])}
    incomplete_target = {'head2': torch.tensor([0])} # missing 'head1'

    result = consistency._constraint_value(
        cons,
        complete_logits,
        incomplete_target,
        ignore_index=255
    )
    assert result is None

    result = consistency._constraint_value(
        cons,
        incomplete_logits,
        complete_target,
        ignore_index=255
    )
    assert result is None

    result = consistency._constraint_value(
        cons,
        incomplete_logits,
        incomplete_target,
        ignore_index=255
    )
    assert result is None

def test_constraint_value_early_exit_no_valid_pixels(mock_constraint):
    '''
    Given: Tensors with all pixels == ignore value.
    When: Private function _constraint_value() is called.
    Then: Returns `None`.
    '''
    cons = mock_constraint(source_head='head1', target_head='head2')
    logits = {
        'head1': torch.randn((2, 3, 8, 8)),
        'head2': torch.randn((2, 3, 8, 8)),
    }
    target = {
        'head1': torch.full((2, 8, 8), 255), # all labels as 255
        'head2': torch.full((2, 8, 8), 255),
    }

    result = consistency._constraint_value(
        cons,
        logits,
        target,
        ignore_index=255
    )
    assert result is None
