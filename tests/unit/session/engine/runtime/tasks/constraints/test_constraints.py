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

'''Unit tests for constraints module (constraints.py).'''

# standard imports
import dataclasses
# third-party imports
import pytest
# local imports
import landseg.session.engine.runtime.tasks.constraints.constraints as constraints


# ----- `CompiledConstraint` dataclass tests
def test_compiled_constraint_dataclass():
    '''
    Given: Field values for a `CompiledConstraint`.
    When: Instantiating `CompiledConstraint`.
    Then: Attributes are initialized correctly and object is frozen.
    '''
    cc = constraints.CompiledConstraint(
        name='rule_1',
        source_head='head_1',
        trigger_val=0,
        target_head='head_2',
        forbidden=(0, 2)
    )

    assert cc.name == 'rule_1'
    assert cc.source_head == 'head_1'
    assert cc.trigger_val == 0
    assert cc.target_head == 'head_2'
    assert cc.forbidden == (0, 2)

    # verify immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        cc.name = 'new_name'


# ----- `compile_constraints` public API tests
def test_compile_constraints_none_returns_none(dataspecs):
    '''
    Given: `mtl_constraints` is `None`.
    When: Calling `compile_constraints`.
    Then: Return `None`.
    '''
    result = constraints.compile_constraints(None, dataspecs)

    assert result is None


def test_compile_constraints_empty_list_returns_none(dataspecs):
    '''
    Given: `mtl_constraints` is an empty list.
    When: Calling `compile_constraints`.
    Then: Return `None`.
    '''
    result = constraints.compile_constraints([], dataspecs)

    assert result is None


def test_compile_constraints_duplicate_names_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: List of constraints with duplicate names.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError` indicating duplicate constraint names.
    '''
    c1 = mock_constraint(name='rule_1')
    c2 = mock_constraint(name='rule_1')

    with pytest.raises(ValueError, match='Duplicated constraints in'):
        constraints.compile_constraints([c1, c2], dataspecs)


def test_compile_constraints_valid(dataspecs, mock_constraint):
    '''
    Given: Valid 1-based constraints.
    When: Calling `compile_constraints`.
    Then: Return list of `CompiledConstraint` objects with 0-based indices.
    '''
    c1 = mock_constraint(
        name='rule_1',
        source_head='head_1',
        trigger_val=1,
        target_head='head_2',
        forbidden=[1, 3]
    )

    result = constraints.compile_constraints([c1], dataspecs)

    assert result is not None
    assert len(result) == 1
    compiled = result[0]  # pylint: disable=unsubscriptable-object
    assert compiled.name == 'rule_1'
    assert compiled.source_head == 'head_1'
    assert compiled.trigger_val == 0
    assert compiled.target_head == 'head_2'
    assert compiled.forbidden == (0, 2)


# ----- `_validate_constraint` internal helper tests
def test_validate_constraint_same_source_and_target_head_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with identical source and target heads.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(source_head='head_1', target_head='head_1')

    with pytest.raises(
        ValueError,
        match='Source and target heads can not be the same'
    ):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_invalid_source_head_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with a non-existent source head.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(source_head='invalid_head')

    with pytest.raises(ValueError, match='Invalid source head'):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_trigger_val_less_than_one_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with `trigger_val` less than 1.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(trigger_val=0)

    with pytest.raises(ValueError, match='trigger_val must be 1-based'):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_trigger_val_out_of_bounds_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with `trigger_val` exceeding class count for source head.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(trigger_val=999)

    with pytest.raises(ValueError, match='Invalid trigger value'):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_invalid_target_head_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with a non-existent target head.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(target_head='invalid_head')

    with pytest.raises(ValueError, match='Invalid target head'):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_empty_forbidden_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with empty forbidden classes list.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(forbidden=[])

    with pytest.raises(
        ValueError,
        match='must contain at least one forbidden class'
    ):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_forbidden_less_than_one_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with a forbidden class less than 1.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(forbidden=[0])

    with pytest.raises(ValueError, match='forbidden classes must be 1-based'):
        constraints.compile_constraints([c], dataspecs)


def test_validate_constraint_forbidden_out_of_bounds_raises(
    dataspecs,
    mock_constraint
):
    '''
    Given: Constraint with a forbidden class exceeding class count for target head.
    When: Calling `compile_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(forbidden=[999])

    with pytest.raises(ValueError, match='Invalid forbidden classes'):
        constraints.compile_constraints([c], dataspecs)
