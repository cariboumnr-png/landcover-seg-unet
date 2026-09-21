# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Unit tests for multihead constraints module (multihead.py).'''

# third-party imports
import pytest
# local imports
import landseg.session.engine.tasks.constraints.multihead as multihead


def test_compile_mtl_constraints_valid(dataspecs, mock_constraint):
    '''
    Given: Valid 1-based constraints.
    When: Calling `compile_mtl_constraints`.
    Then: Return list of `CompiledMTLConstraint` objects with 0-based
        indices.
    '''
    c1 = mock_constraint()
    result = multihead.compile_mtl_constraints([c1], dataspecs)

    assert result is not None
    assert len(result) == 1
    compiled = result[0]  # pylint: disable=unsubscriptable-object
    # default values; see mock_constraint fixture
    assert compiled.name == 'rule_1'
    assert compiled.source_head == 'head_1'
    assert compiled.trigger_val == 0 # 1 -> 0
    assert compiled.target_head == 'head_2'
    assert compiled.forbidden == (1,) #  [2] -> (1,)


def test_compile_mtl_constraints_duplicated_raises(
    dataspecs, mock_constraint
):
    '''
    Given: List of constraints with duplicate names.
    When: Calling `compile_mtl_constraints`.
    Then: Raise `ValueError` indicating duplicate constraint names.
    '''
    c1 = mock_constraint(name='rule_1')
    c2 = mock_constraint(name='rule_1')

    with pytest.raises(ValueError, match='Duplicated constraints in'):
        multihead.compile_mtl_constraints([c1, c2], dataspecs)


@pytest.mark.parametrize('inputs', (None, []))
def test_compile_mtl_constraints_none_or_empty_returns_none(
    inputs, dataspecs
):
    '''
    Given: `mtl_constraints` is `None` or an empty list `[]`.
    When: Calling `compile_mtl_constraints`.
    Then: Return `None`.
    '''
    result = multihead.compile_mtl_constraints(inputs, dataspecs)

    assert result is None


def test_validate_constraint_raises_value_error(dataspecs, mock_constraint):
    '''
    Given: Constraint with various invalid inputs.
    When: Calling `compile_mtl_constraints`.
    Then: Raise `ValueError`.
    '''
    c = mock_constraint(source_head='head_1', target_head='head_1')
    with pytest.raises(ValueError, match='heads can not be the same'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(source_head='invalid_head')
    with pytest.raises(ValueError, match='invalid source head'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(trigger_val=0)
    with pytest.raises(ValueError, match='trigger value must be 1-based'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(trigger_val=999)
    with pytest.raises(ValueError, match='trigger value is out of range'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(target_head='invalid_head')
    with pytest.raises(ValueError, match='invalid target head'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(forbidden=[])
    with pytest.raises(ValueError, match='empty forbidden class list'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(forbidden=[0])
    with pytest.raises(ValueError, match='forbidden classes must be 1-based'):
        multihead.compile_mtl_constraints([c], dataspecs)

    c = mock_constraint(forbidden=[999])
    with pytest.raises(ValueError, match='out of range forbidden classes'):
        multihead.compile_mtl_constraints([c], dataspecs)
