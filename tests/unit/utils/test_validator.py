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

'''Unit tests for standalone validation primitives in validator.py.'''

# standard imports
import pathlib
import types
# third-party imports
import pytest
# local imports
from landseg.utils.validator import (
    ValidationError,
    require_file,
    require_mapping,
    require_mapping_str_v,
    require_numeric,
    require_numeric_max,
    require_numeric_min,
    require_numeric_range,
    require_numeric_sequence,
    require_sequence,
    require_string,
    require_string_sequence,
    require_types,
    require_value_from,
)


# ----- `require_types` validation
@pytest.mark.parametrize(
    ('value', 'required_type', 'expected'),
    [
        (10, int, 10),
        ('test', str, 'test'),
        (3.14, float, 3.14),
        (5, (int, float), 5),
    ],
)
def test_require_types_accepts_valid_types(value, required_type, expected):
    '''Given a value matching required type(s), return the value.'''
    assert require_types(value, required_type) == expected


def test_require_types_rejects_invalid_type():
    '''Given a value not matching required type, raise ValidationError.'''
    with pytest.raises(ValidationError, match='Expected type'):
        require_types('not_int', int)


# ----- `require_numeric` validation
@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (0, 0),
        (12, 12),
        (-3, -3),
        (1.5, 1.5),
        (-2.75, -2.75),
    ],
)
def test_require_numeric_accepts_numbers(value, expected):
    '''Given a numeric value, when validated, then return the value.'''
    assert require_numeric(value) == expected


@pytest.mark.parametrize(
    ('value', 'required_type', 'expected'),
    [
        (3, int, 3),
        (-2, int, -2),
        (1.25, float, 1.25),
        (-0.5, float, -0.5),
    ],
)
def test_require_numeric_enforces_requested_type(
    value,
    required_type,
    expected,
):
    '''Given a value of the requested type, when validated, then return it.'''
    assert require_numeric(value, required_type) == expected


@pytest.mark.parametrize(
    ('value', 'required_type'),
    [
        (1.5, int),
        (1, float),
        ('1', None),
        (None, None),
        ([], None),
    ],
)
def test_require_numeric_rejects_invalid_types(value, required_type):
    '''Given an invalid numeric type, when validated, then raise an error.'''
    with pytest.raises(ValidationError):
        require_numeric(value, required_type)


# ----- `require_numeric_min` validation
@pytest.mark.parametrize(
    ('value', 'minimum', 'required_type', 'expected'),
    [
        (0, 0, None, 0),
        (5, 3, None, 5),
        (-1.5, -2.0, None, -1.5),
        (4, 4, int, 4),
        (2.5, 2.0, float, 2.5),
    ],
)
def test_require_numeric_min_accepts_values_at_or_above_minimum(
    value,
    minimum,
    required_type,
    expected,
):
    '''Given a value meeting the minimum, when checked, then return it.'''
    assert (
        require_numeric_min(value, minimum, required_type)
        == expected
    )


@pytest.mark.parametrize(
    ('value', 'minimum', 'required_type'),
    [
        (2, 3, None),
        (-2.0, -1.0, None),
        (3, 4, int),
        (1.5, 2.0, float),
    ],
)
def test_require_numeric_min_rejects_values_below_minimum(
    value,
    minimum,
    required_type,
):
    '''Given a value below the minimum, when checked, then raise an error.'''
    with pytest.raises(ValidationError, match='minimum'):
        require_numeric_min(value, minimum, required_type)


# ----- `require_numeric_max` validation
@pytest.mark.parametrize(
    ('value', 'maximum', 'required_type', 'expected'),
    [
        (0, 0, None, 0),
        (3, 5, None, 3),
        (-1.5, -1.0, None, -1.5),
        (4, 4, int, 4),
        (2.5, 3.0, float, 2.5),
    ],
)
def test_require_numeric_max_accepts_values_at_or_below_maximum(
    value,
    maximum,
    required_type,
    expected,
):
    '''Given a value meeting the maximum, when checked, then return it.'''
    assert (
        require_numeric_max(value, maximum, required_type)
        == expected
    )


@pytest.mark.parametrize(
    ('value', 'maximum', 'required_type'),
    [
        (4, 3, None),
        (0.5, 0.0, None),
        (5, 4, int),
        (3.5, 3.0, float),
    ],
)
def test_require_numeric_max_rejects_values_above_maximum(
    value,
    maximum,
    required_type,
):
    '''Given a value above the maximum, when checked, then raise an error.'''
    with pytest.raises(ValidationError, match='maximum'):
        require_numeric_max(value, maximum, required_type)


# ----- `require_numeric_range` validation
@pytest.mark.parametrize(
    ('value', 'bounds', 'required_type', 'expected'),
    [
        (0, (0, 10), None, 0),
        (10, (0, 10), None, 10),
        (5, (0, 10), int, 5),
        (2.5, (1.0, 3.0), float, 2.5),
        (2, (1, 3), int, 2),
        (-1.5, (-2.0, 0.0), None, -1.5),
    ],
)
def test_require_numeric_range_accepts_values_within_closed_bounds(
    value,
    bounds,
    required_type,
    expected,
):
    '''Given a value within inclusive bounds, when checked, then return it.'''
    assert (
        require_numeric_range(value, bounds, required_type)
        == expected
    )


@pytest.mark.parametrize(
    ('value', 'bounds', 'required_type'),
    [
        (-1, (0, 10), None),
        (11, (0, 10), None),
        (0.5, (1.0, 3.0), float),
        (4.5, (1.0, 3.0), float),
    ],
)
def test_require_numeric_range_rejects_values_outside_bounds(
    value,
    bounds,
    required_type,
):
    '''Given a value outside inclusive bounds, when checked, then raise.'''
    with pytest.raises(ValidationError):
        require_numeric_range(value, bounds, required_type)


# ----- `require_string` validation
@pytest.mark.parametrize(
    'value',
    [
        'hello',
        'EPSG:2958',
        ' ',
        '0',
    ],
)
def test_require_string_accepts_nonempty_strings(value):
    '''Given a nonempty string, when validated, then return it unchanged.'''
    assert require_string(value) == value


@pytest.mark.parametrize(
    'value',
    [
        1,
        1.5,
        None,
        [],
        {},
    ],
)
def test_require_string_rejects_non_string_values(value):
    '''Given a non-string value, when validated, then raise TypeError.'''
    with pytest.raises(TypeError):
        require_string(value)


def test_require_string_rejects_empty_string():
    '''Given an empty string, when validated, then raise ValueError.'''
    with pytest.raises(ValueError):
        require_string('')


# ----- `require_value_from` validation
@pytest.mark.parametrize(
    ('value', 'allowed', 'required_type'),
    [
        ('rgb', ('rgb', 'all'), str),
        ('all', ('rgb', 'all'), str),
        (1, (1, 2, 3), int),
        (2.5, (1.5, 2.5), float),
    ],
)
def test_require_value_from_accepts_allowed_values(
    value,
    allowed,
    required_type,
):
    '''Given an allowed value of the requested type, when checked, return it.'''
    assert (
        require_value_from(value, allowed, required_type)
        == value
    )


@pytest.mark.parametrize(
    ('value', 'allowed', 'required_type'),
    [
        ('dem', ('rgb', 'all'), str),
        (4, (1, 2, 3), int),
        (1.5, (1.0, 2.0), float),
    ],
)
def test_require_value_from_rejects_values_outside_allowed_pool(
    value,
    allowed,
    required_type,
):
    '''Given a value outside the allowed pool, when checked, then raise.'''
    with pytest.raises(ValidationError):
        require_value_from(value, allowed, required_type)


def test_require_value_from_rejects_wrong_type():
    '''Given a value of the wrong type, when checked, then raise an error.'''
    with pytest.raises(ValidationError):
        require_value_from(1, ('1', '2'), str)


def test_require_value_from_rejects_string_as_allowed_sequence():
    '''Given a string pool, when checked, then reject it as a sequence.'''
    with pytest.raises(ValidationError):
        require_value_from('a', 'abc', str)


# ----- `require_mapping` validation
def test_require_mapping_returns_dictionary_copy():
    '''Given a nonempty mapping, when validated, then return a dict copy.'''
    source = {'a': 1, 'b': 2}
    result = require_mapping(source)

    assert result == source
    assert isinstance(result, dict)
    assert result is not source


def test_require_mapping_accepts_mapping_proxy():
    '''Given a read-only mapping, when validated, then return a dict.'''
    source = types.MappingProxyType({'a': 1})
    result = require_mapping(source)

    assert result == {'a': 1}
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    'value',
    [
        None,
        1,
        'abc',
        [],
    ],
)
def test_require_mapping_rejects_non_mappings(value):
    '''Given a non-mapping value, when validated, then raise TypeError.'''
    with pytest.raises(TypeError):
        require_mapping(value)


def test_require_mapping_rejects_empty_mapping():
    '''Given an empty mapping, when validated, then raise ValueError.'''
    with pytest.raises(ValueError):
        require_mapping({})


def test_require_mapping_str_v_returns_validated_copy():
    '''Given a mapping with string values, when checked, then return a copy.'''
    source = {'drainage': 'drainage.tif', 'geology': 'geology.tif'}
    result = require_mapping_str_v(source)

    assert result == source
    assert result is not source


def test_require_mapping_str_v_rejects_non_string_value():
    '''Given a non-string mapping value, when checked, then raise TypeError.'''
    with pytest.raises(TypeError):
        require_mapping_str_v({'a': 'valid', 'b': 2})


def test_require_mapping_str_v_rejects_empty_string_value():
    '''Given an empty mapping value, when checked, then raise ValueError.'''
    with pytest.raises(ValueError):
        require_mapping_str_v({'a': ''})


# ----- `require_sequence` validation
@pytest.mark.parametrize(
    'value',
    [
        [1, 2],
        (1, 2),
        range(3),
    ],
)
def test_require_sequence_returns_tuple(value):
    '''Given a non-string sequence, when validated, then return a tuple.'''
    result = require_sequence(value)

    assert result == tuple(value)
    assert isinstance(result, tuple)


@pytest.mark.parametrize(
    'value',
    [
        'abc',
        b'abc',
        bytearray(b'abc'),
        123,
        None,
        {'a': 1},
        {1, 2},
    ],
)
def test_require_sequence_rejects_non_sequences(value):
    '''Given an invalid sequence input, when validated, then raise an error.'''
    with pytest.raises(ValidationError):
        require_sequence(value)


@pytest.mark.parametrize(
    ('value', 'required_type', 'length', 'expected'),
    [
        ([1, 2, 3], None, None, (1, 2, 3)),
        ((1, 2), int, 2, (1, 2)),
        ([1.0, 2.0], float, 2, (1.0, 2.0)),
        ([], None, 0, ()),
    ],
)
def test_require_numeric_sequence_accepts_valid_sequences(
    value,
    required_type,
    length,
    expected,
):
    '''Given a valid numeric sequence, when checked, then return a tuple.'''
    assert (
        require_numeric_sequence(value, required_type, length)
        == expected
    )


@pytest.mark.parametrize(
    ('value', 'required_type', 'length'),
    [
        ([1, '2'], None, None),
        ([1.0, 2.0], int, None),
        ([1, 2], float, None),
        ([1, 2], int, 3),
        ([1.0], float, 2),
    ],
)
def test_require_numeric_sequence_rejects_invalid_sequences(
    value,
    required_type,
    length,
):
    '''Given an invalid numeric sequence, when checked, then raise an error.'''
    with pytest.raises(ValidationError):
        require_numeric_sequence(value, required_type, length)


@pytest.mark.parametrize(
    ('value', 'length', 'expected'),
    [
        (['rgb', 'nir'], None, ('rgb', 'nir')),
        (('dem', 'hag'), 2, ('dem', 'hag')),
        ([], 0, ()),
    ],
)
def test_require_string_sequence_accepts_valid_sequences(
    value,
    length,
    expected,
):
    '''Given a valid string sequence, when checked, then return a tuple.'''
    assert require_string_sequence(value, length) == expected


@pytest.mark.parametrize(
    ('value', 'length'),
    [
        (['rgb', 1], None),
        (['rgb', 'nir'], 1),
        ([''], None),
    ],
)
def test_require_string_sequence_rejects_invalid_sequences(value, length):
    '''Given an invalid string sequence, when checked, then raise an error.'''
    with pytest.raises((TypeError, ValueError, ValidationError)):
        require_string_sequence(value, length)


# ----- `require_file` validation
def test_require_file_returns_path_for_existing_file(tmp_path):
    '''Given an existing file, when validated, then return its Path.'''
    source = tmp_path / 'input.tif'
    source.write_bytes(b'test')

    result = require_file(str(source))

    assert result == source
    assert isinstance(result, pathlib.Path)
    assert result.is_file()


def test_require_file_accepts_path_string(tmp_path):
    '''Given a valid file path string, when checked, then return a Path.'''
    source = tmp_path / 'config.yaml'
    source.write_text('key: value\n', encoding='utf-8')

    assert require_file(str(source)) == source


def test_require_file_accepts_path_object(tmp_path):
    '''Given a valid Path instance, when checked, then return a Path.'''
    source = tmp_path / 'config.yaml'
    source.write_text('key: value\n', encoding='utf-8')

    assert require_file(source) == source


def test_require_file_rejects_missing_file(tmp_path):
    '''Given a nonexistent file, when validated, then raise ValidationError.'''
    source = tmp_path / 'missing.tif'

    with pytest.raises(ValidationError, match='Not a valid file'):
        require_file(str(source))


def test_require_file_rejects_directory(tmp_path):
    '''Given a directory path, when validated, then raise ValidationError.'''
    with pytest.raises(ValidationError, match='Not a valid file'):
        require_file(str(tmp_path))


def test_require_file_rejects_missing_path_when_required():
    '''Given no path and missing_ok false, when checked, then raise an error.'''
    with pytest.raises(ValidationError, match='Path not provided'):
        require_file(None)


def test_require_file_accepts_missing_path_when_optional():
    '''Given no path and missing_ok true, when checked, then return None.'''
    assert require_file(None, missing_ok=True) is None
