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

# pylint: disable=protected-access

'''
Unit tests for `landseg.configs.schema.base`.
'''

# standard imports
import dataclasses
# third-party imports
import pytest
# local imports
import landseg.configs.schema.base as base


# ----- `BaseConfigSection` file validation tests
def test_file_exists_and_require_file(tmp_path):
    '''
    Given: Existing and non-existent file paths.
    When: `file_exists` and `require_file` helpers are called.
    Then: Return existence flags or raise `ConfigValidationError`.
    '''
    dummy_file = tmp_path / 'test.txt'
    dummy_file.write_text('content')

    # require_file passes silently for existing file or None / empty
    base.BaseConfigSection.require_file(str(dummy_file), 'dummy')
    base.BaseConfigSection.require_file(None, 'none_path')
    base.BaseConfigSection.require_file('', 'empty_path')

    # require_file raises ConfigValidationError for missing path
    with pytest.raises(base.ConfigValidationError, match='is not file'):
        base.BaseConfigSection.require_file(
            str(tmp_path / 'missing.txt'), 'missing'
        )


# ----- `BaseConfigSection` type and range validation tests
def test_require_type_range_validation():
    '''
    Given: Values tested against types, numeric ranges, and literal sets.
    When: `require_type_range` is executed.
    Then: Pass valid inputs or raise ValueError for invalid inputs.
    '''
    # none value returns early
    base.BaseConfigSection.require_type_range(None, 'val', int)

    # valid numeric values within range
    base.BaseConfigSection.require_type_range(5, 'val', int, (0, 10))
    base.BaseConfigSection.require_type_range(0.5, 'val', float, (0.0, 1.0))

    # wrong type raises ValueError
    with pytest.raises(ValueError, match='Expected type'):
        base.BaseConfigSection.require_type_range('string', 'val', int)

    # out of bounds lower
    with pytest.raises(ValueError, match='must be within'):
        base.BaseConfigSection.require_type_range(-1, 'val', int, (0, 10))

    # out of bounds upper
    with pytest.raises(ValueError, match='must be within'):
        base.BaseConfigSection.require_type_range(11, 'val', int, (0, 10))

    # valid literal string
    base.BaseConfigSection.require_type_range(
        'adam', 'opt', str, ['adam', 'sgd']
    )

    # invalid literal string
    with pytest.raises(ValueError, match='must be in the following'):
        base.BaseConfigSection.require_type_range(
            'rmsprop', 'opt', str, ['adam', 'sgd']
        )

    # empty string raises ValueError
    with pytest.raises(ValueError, match='must be non-empty'):
        base.BaseConfigSection.require_type_range('  ', 'opt', str)


# ----- `BaseConfigSection` attribute validation and serialization tests
def test_base_config_section_attr_validation_and_as_dict():
    '''
    Given: A custom dataclass section subclassing `BaseConfigSection`.
    When: Calling `require_attr_type_range`, `as_dict`, and `validate`.
    Then: Validate attributes and serialize dataclass to dictionary.
    '''
    @dataclasses.dataclass
    class DummySection(base.BaseConfigSection):
        epoch: int = 10
        mode: str = 'train'

        def validate(self) -> None:
            self.require_attr_type_range('epoch', int, (1, 100))
            self.require_attr_type_range('mode', str, ['train', 'val'])

    section = DummySection()
    section.validate()
    assert section.as_dict == {'epoch': 10, 'mode': 'train'}

    # default base class validate is no-op
    base_sec = base.BaseConfigSection()
    assert base_sec.validate() is None

    # invalid attribute value raises ConfigValidationError
    invalid_sec = DummySection(epoch=-5)
    with pytest.raises(base.ConfigValidationError):
        invalid_sec.validate()
