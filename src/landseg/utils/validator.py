# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''
Standalone validation primitives and composed validation helpers.

The module has no dependencies outside the Python standard library.

Exception conventions:
    TypeError: an input has an invalid type.
    ValueError: an input value or constraint is invalid.
    ValidationError: an attribute/configuration validation failure.
'''

# standard imports
import collections.abc
import pathlib
import typing


T = typing.TypeVar('T', int, float)
V = typing.TypeVar('V')


# ----- typing aliases
NumericRange: typing.TypeAlias = tuple[int | float, int | float]

# ----- public classes
class ValidationError(Exception):
    '''Raised when configuration or attribute validation fails.'''


# ----- public functions
def require_types(
    value: object,
    name: str,
    required_type: type[V] | tuple[type[V], ...]
) -> V:
    '''Require a type or a tuple of types.'''
    if not isinstance(value, required_type):
        if isinstance(required_type, tuple):
            required = ', '.join(t.__name__ for t in required_type)
        else:
            required = required_type.__name__

        raise ValidationError(
            f'Expected type: {required} for {name}, '
            f'got {value} with type: {type(value)} instead'
        )
    return value


# numerics
@typing.overload
def require_numeric(
    value: object,
    value_name: str,
    require_type: None = None,
) -> float | int: ...


@typing.overload
def require_numeric(
    value: object,
    value_name: str,
    require_type: type[T],
) -> T: ...


def require_numeric(
    value: object,
    value_name: str,
    require_type: type[T] | None = None,
) -> T | float | int:
    '''Require an integer and a float.'''
    if require_type is float:
        if not isinstance(value, float):
            raise ValidationError(f'{value_name} not an float')
        return value
    if require_type is int:
        if not isinstance(value, int):
            raise ValidationError(f'{value_name} not an integer')
        return value
    if not isinstance(value, (int, float)):
        raise ValidationError(f'{value_name} not a number')
    return value


@typing.overload
def require_numeric_min(
    value: object,
    value_name: str,
    value_min: float | int,
    require_type: None = None,
) -> float | int: ...


@typing.overload
def require_numeric_min(
    value: object,
    value_name: str,
    value_min: float | int,
    require_type: type[T],
) -> T: ...


def require_numeric_min(
    value: object,
    value_name: str,
    value_min: float | int,
    require_type: type[T] | None = None,
) -> T | float | int:
    '''Require a number >= minimum.'''
    _value_min = require_numeric(value_min, 'minimum')
    _value = require_numeric(value, value_name, require_type)
    if _value < _value_min:
        raise ValidationError(f'{value_name}: {_value} < minimum {_value_min}')
    return _value


@typing.overload
def require_numeric_max(
    value: object,
    value_name: str,
    value_max: float | int,
    require_type: None = None,
) -> float | int: ...


@typing.overload
def require_numeric_max(
    value: object,
    value_name: str,
    value_max: float | int,
    require_type: type[T],
) -> T: ...


def require_numeric_max(
    value: object,
    value_name: str,
    value_max: float | int,
    require_type: type[T] | None = None,
) -> T | float | int:
    '''Require a number <= maximum.'''
    _value_max = require_numeric(value_max, 'maximum')
    _value = require_numeric(value, value_name, require_type)
    if _value > _value_max:
        raise ValidationError(f'{value_name}: {_value} > maximum {_value_max}')
    return _value


@typing.overload
def require_numeric_range(
    value: object,
    value_name: str,
    require_range: NumericRange,
    require_type: None = None,
) -> float | int: ...


@typing.overload
def require_numeric_range(
    value: object,
    value_name: str,
    require_range: NumericRange,
    require_type: type[T],
) -> T: ...


def require_numeric_range(
    value: object,
    value_name: str,
    require_range: NumericRange,
    require_type: type[T] | None = None,
) -> T | float | int:
    '''Require a number within closed range (min <= x <=max).'''
    _value = require_numeric(value, value_name, require_type)
    _require_range = require_numeric_sequence(require_range, 'range', k=2)
    mmin, mmax = _require_range
    require_numeric_min(_value, value_name, mmin, require_type)
    require_numeric_max(_value, value_name, mmax, require_type)
    return _value


# strings
def require_string(
    value: object,
    value_name: str,
) -> str:
    '''Require a non-empty string; return as a str.'''
    if not isinstance(value, str):
        raise TypeError(f'{value_name} not a string: {type(value).__name__}')
    if not value:
        raise ValueError(f'{value_name} is an empty string')
    return value


# conditions
def require_value_from(
    value: object,
    value_name: str,
    require_from: typing.Sequence[V],
    require_type: type[V]
) -> V:
    '''Require an object from a pool, e.g., allowed literals.'''
    if not isinstance(value, require_type):
        actual_type = type(value).__name__
        raise ValidationError(f'{value_name} not required type: {actual_type}')
    _range = require_sequence(require_from, 'required_range')
    if value not in _range:
        raise ValidationError(f'{value_name} not in range: {require_from}')
    return value


# mappings
def require_mapping(
    value: object,
    value_name: str,
) -> dict:
    '''Require a non-empty mapping; return as a dict.'''
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError(f'{value_name} not a mapping: {type(value).__name__}')
    if not value:
        raise ValueError(f'{value_name} is an empty dict')
    return dict(value)


def require_mapping_str_v(
    value: object,
    value_name: str,
) -> dict[typing.Any, str]:
    '''Require mapping with string values; return as a dict.'''
    _value = require_mapping(value, value_name)
    _validated: dict[typing.Any, str] = {}
    for k, v in _value.items():
        _validated[k] = require_string(v, f'value@key{k}')
    return _validated


# sequences
def require_sequence(
    value: object,
    value_name: str,
    k: int | None = None,
) -> tuple:
    '''Require a sequence of objects; return as a tuple.'''
    if not (
        isinstance(value, collections.abc.Sequence) and
        not isinstance(value, (str, bytes, bytearray))
    ):
        raise ValidationError(f'{value_name} not a sequence of objects.')
    if k is not None and len(value) != k:
        raise ValidationError(f'{value_name}: len={len(value)} != required {k}')
    return tuple(value)


@typing.overload
def require_numeric_sequence(
    value: object,
    value_name: str,
    require_type: None = None,
    k: int | None = None,
) -> tuple[int | float, ...]: ...


@typing.overload
def require_numeric_sequence(
    value: object,
    value_name: str,
    require_type: type[T],
    k: int | None = None,
) -> tuple[T, ...]: ...


def require_numeric_sequence(
    value: object,
    value_name: str,
    require_type: type[T] | None = None,
    k: int | None = None,
) -> tuple[T, ...] | tuple[int | float, ...]:
    '''Require a sequence of numbers of optional specified length (k).'''
    _value = require_sequence(value, value_name, k=k)
    for i, v in enumerate(_value):
        require_numeric(v, f'value_{i}', require_type)
    return _value


def require_string_sequence(
    value: object,
    value_name: str,
    k: int | None = None,
) -> tuple[str, ...]:
    '''Require a sequence of strings of optional specified length (k).'''
    _value = require_sequence(value, value_name, k=k)
    for i, v in enumerate(_value):
        require_string(v, f'value_{i}')
    return _value


# files
@typing.overload
def require_file(
    path: str | pathlib.Path | None,
    file_name: str,
    *,
    missing_ok: typing.Literal[False] = False,
) -> pathlib.Path: ...


@typing.overload
def require_file(
    path: str | pathlib.Path | None,
    file_name: str,
    *,
    missing_ok: typing.Literal[True],
) -> pathlib.Path | None: ...


@typing.overload
def require_file(
    path: str | pathlib.Path | None,
    file_name: str,
    *,
    missing_ok: bool = False,
) -> pathlib.Path | None: ...


def require_file(
    path: str | pathlib.Path | None,
    file_name: str,
    *,
    missing_ok: bool = False,
) -> pathlib.Path | None:
    '''Require an existing regular file; return as a `Path` object.'''
    if path is None:
        if missing_ok:
            return None
        raise ValidationError(f'Path not provided for {file_name}')
    try:
        _path = pathlib.Path(path)
    except (TypeError, ValueError) as e:
        raise ValidationError(f'{file_name} not a valid path: {path}') from e
    if not pathlib.Path(_path).is_file():
        raise ValidationError(f'{file_name} not a valid file: {path}')
    return _path
