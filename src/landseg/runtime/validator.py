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
    - `TypeError`: Input has an invalid type.
    - `ValueError`: Input value or constraint is invalid.
    - `ValidationError`: Attribute or configuration validation failure.

Public APIs:
    - `NumericRange`: Inclusive numeric `(min, max)` bounds alias.
    - `ValidationError`: Raised when configuration validation fails.
    - `require_dir`: Require an existing or writable directory path.
    - `require_file`: Require an existing regular file path.
    - `require_finite`: Require a finite numeric value.
    - `require_mapping`: Require a non-empty mapping as a dictionary.
    - `require_mapping_str_v`: Require a mapping with string values.
    - `require_numeric`: Require an integer or floating-point number.
    - `require_numeric_max`: Require a number at or below maximum.
    - `require_numeric_min`: Require a number at or above minimum.
    - `require_numeric_range`: Require a number within closed bounds.
    - `require_numeric_sequence`: Require a sequence of numeric values.
    - `require_sequence`: Require a sequence of objects as a tuple.
    - `require_string`: Require a non-empty string value.
    - `require_string_sequence`: Require a sequence of strings.
    - `require_types`: Require a value matching expected type(s).
    - `require_value_from`: Require a typed value from allowed choices.
'''

# standard imports
import collections.abc
import math
import numbers
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
    required_type: type[V] | tuple[type[V], ...]
) -> V:
    '''
    Require a value matching a target type or tuple of types.

    Args:
        value:
            Input object to validate.
        required_type:
            Expected type or tuple of allowed types.

    Returns:
        V:
            Validated input object cast to the expected type.

    Raises:
        ValidationError:
            If `value` is not an instance of `required_type`.
    '''
    if not isinstance(value, required_type):
        if isinstance(required_type, tuple):
            required = ', '.join(t.__name__ for t in required_type)
        else:
            required = required_type.__name__

        raise ValidationError(
            f'Expected type: {required}, '
            f'got {value} with type: {type(value)} instead'
        )
    return value


# numerics
@typing.overload
def require_numeric(
    value: object,
    require_type: None = None,
    *,
    finite: bool = False,
) -> float | int: ...


@typing.overload
def require_numeric(
    value: object,
    require_type: type[T],
    *,
    finite: bool = False,
) -> T: ...


def require_numeric(
    value: object,
    require_type: type[T] | None = None,
    *,
    finite: bool = False,
) -> T | float | int:
    '''
    Require an integer or floating-point number.

    Args:
        value:
            Input object to validate as numeric (excluding `bool`).
        require_type:
            Optional exact numeric type (`int` or `float`) to enforce.
        finite:
            Whether to reject non-finite values (`NaN` or `inf`).

    Returns:
        T | float | int:
            Validated numeric value.

    Raises:
        ValidationError:
            If `value` is a boolean, does not match the numeric type,
            or is non-finite when `finite` is `True`.
    '''
    if isinstance(value, bool):
        raise ValidationError('Expected a number: got bool')
    if require_type is float:
        if not (
            isinstance(value, float)
            or (
                isinstance(value, numbers.Real)
                and not isinstance(value, numbers.Integral)
            )
        ):
            raise ValidationError(f'Expected a float: got {type(value)}')
    elif require_type is int:
        if not isinstance(value, (int, numbers.Integral)):
            raise ValidationError(f'Expected an integer: got {type(value)}')
    else:
        if not isinstance(value, (int, float, numbers.Real)):
            raise ValidationError(f'Expected a number: got {type(value)}')

    if finite and not math.isfinite(typing.cast(float | int, value)):
        raise ValidationError(f'Expected a finite number: got {value}')
    return value  # type: ignore[return-value]


@typing.overload
def require_finite(
    value: object,
    require_type: None = None,
) -> float | int: ...


@typing.overload
def require_finite(
    value: object,
    require_type: type[T],
) -> T: ...


def require_finite(
    value: object,
    require_type: type[T] | None = None,
) -> T | float | int:
    '''
    Require a finite integer or floating-point number.

    Args:
        value:
            Input object to validate as a finite number.
        require_type:
            Optional exact numeric type (`int` or `float`) to enforce.

    Returns:
        T | float | int:
            Validated finite numeric value.

    Raises:
        ValidationError:
            If `value` fails numeric type or finiteness checks.
    '''
    return require_numeric(value, require_type, finite=True)


@typing.overload
def require_numeric_min(
    value: object,
    value_min: float | int,
    require_type: None = None,
    *,
    finite: bool = False,
) -> float | int: ...


@typing.overload
def require_numeric_min(
    value: object,
    value_min: float | int,
    require_type: type[T],
    *,
    finite: bool = False,
) -> T: ...


def require_numeric_min(
    value: object,
    value_min: float | int,
    require_type: type[T] | None = None,
    *,
    finite: bool = False,
) -> T | float | int:
    '''
    Require a numeric value greater than or equal to a minimum.

    Args:
        value:
            Input object to validate.
        value_min:
            Inclusive lower bound.
        require_type:
            Optional exact numeric type (`int` or `float`) to enforce.
        finite:
            Whether to reject non-finite values.

    Returns:
        T | float | int:
            Validated numeric value meeting the lower bound.

    Raises:
        ValidationError:
            If `value` or `value_min` is invalid, or
            `value < value_min`.
    '''
    _value_min = require_numeric(value_min, finite=finite)
    _value = require_numeric(value, require_type, finite=finite)
    if _value < _value_min:
        raise ValidationError(f'{_value} < minimum {_value_min}')
    return _value


@typing.overload
def require_numeric_max(
    value: object,
    value_max: float | int,
    require_type: None = None,
    *,
    finite: bool = False,
) -> float | int: ...


@typing.overload
def require_numeric_max(
    value: object,
    value_max: float | int,
    require_type: type[T],
    *,
    finite: bool = False,
) -> T: ...


def require_numeric_max(
    value: object,
    value_max: float | int,
    require_type: type[T] | None = None,
    *,
    finite: bool = False,
) -> T | float | int:
    '''
    Require a numeric value less than or equal to a maximum.

    Args:
        value:
            Input object to validate.
        value_max:
            Inclusive upper bound.
        require_type:
            Optional exact numeric type (`int` or `float`) to enforce.
        finite:
            Whether to reject non-finite values.

    Returns:
        T | float | int:
            Validated numeric value meeting the upper bound.

    Raises:
        ValidationError:
            If `value` or `value_max` is invalid, or
            `value > value_max`.
    '''
    _value_max = require_numeric(value_max, finite=finite)
    _value = require_numeric(value, require_type, finite=finite)
    if _value > _value_max:
        raise ValidationError(f'{_value} > maximum {_value_max}')
    return _value


@typing.overload
def require_numeric_range(
    value: object,
    require_range: NumericRange,
    require_type: None = None,
    *,
    finite: bool = False,
) -> float | int: ...


@typing.overload
def require_numeric_range(
    value: object,
    require_range: NumericRange,
    require_type: type[T],
    *,
    finite: bool = False,
) -> T: ...


def require_numeric_range(
    value: object,
    require_range: NumericRange,
    require_type: type[T] | None = None,
    *,
    finite: bool = False,
) -> T | float | int:
    '''
    Require a number within an inclusive range (`min <= x <= max`).

    Args:
        value:
            Input object to validate.
        require_range:
            Two-element `(min, max)` inclusive bounds tuple.
        require_type:
            Optional exact numeric type (`int` or `float`) to enforce.
        finite:
            Whether to reject non-finite values.

    Returns:
        T | float | int:
            Validated numeric value within the closed interval.

    Raises:
        ValidationError:
            If `value` or `require_range` is invalid, or `value` lies
            outside the bounds.
    '''
    _value = require_numeric(value, require_type, finite=finite)
    _require_range = require_numeric_sequence(require_range, k=2)
    mmin, mmax = _require_range
    require_numeric_min(_value, mmin, require_type, finite=finite)
    require_numeric_max(_value, mmax, require_type, finite=finite)
    return _value


# strings
def require_string(
    value: object,
) -> str:
    '''
    Require a non-empty string value.

    Args:
        value:
            Input object to validate as a non-empty string.

    Returns:
        str:
            Validated non-empty string.

    Raises:
        TypeError:
            If `value` is not a `str`.
        ValueError:
            If `value` is an empty string.
    '''
    if not isinstance(value, str):
        raise TypeError(f'Expected a string, got {type(value).__name__}')
    if not value:
        raise ValueError('String must not be empty')
    return value


# conditions
def require_value_from(
    value: object,
    require_from: typing.Sequence[V],
    require_type: type[V]
) -> V:
    '''
    Require a typed value belonging to an allowed sequence of choices.

    Args:
        value:
            Input object to validate.
        require_from:
            Sequence of allowed values.
        require_type:
            Expected runtime type of `value`.

    Returns:
        V:
            Validated value from `require_from`.

    Raises:
        ValidationError:
            If `value` has the wrong type, `require_from` is not a
            sequence, or `value` is not in `require_from`.
    '''
    if not isinstance(value, require_type):
        actual_type = type(value).__name__
        raise ValidationError(
            f'Expected type {require_type.__name__}, got {actual_type}'
        )
    _range = require_sequence(require_from)
    if value not in _range:
        raise ValidationError(f'Value not in allowed range: {require_from}')
    return value


# mappings
def require_mapping(
    value: object,
) -> dict:
    '''
    Require a non-empty mapping and return a shallow dictionary copy.

    Args:
        value:
            Input object to validate as a non-empty mapping.

    Returns:
        dict:
            Shallow `dict` copy of the validated mapping.

    Raises:
        TypeError:
            If `value` is not a `collections.abc.Mapping`.
        ValueError:
            If `value` is empty.
    '''
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError(f'Expected a mapping, got {type(value).__name__}')
    if not value:
        raise ValueError('Mapping must not be empty')
    return dict(value)


def require_mapping_str_v(
    value: object,
) -> dict[typing.Any, str]:
    '''
    Require a non-empty mapping whose values are non-empty strings.

    Args:
        value:
            Input object to validate as a string-valued mapping.

    Returns:
        dict[typing.Any, str]:
            Validated dictionary mapping keys to non-empty strings.

    Raises:
        TypeError:
            If `value` is not a mapping or any value is not a `str`.
        ValueError:
            If `value` is empty or any string value is empty.
    '''
    _value = require_mapping(value)
    _validated: dict[typing.Any, str] = {}
    for k, v in _value.items():
        _validated[k] = require_string(v)
    return _validated


# sequences
def require_sequence(
    value: object,
    k: int | None = None,
) -> tuple:
    '''
    Require a non-string sequence of objects and return as a tuple.

    Args:
        value:
            Input object to validate (excluding `str`, `bytes`, and
            `bytearray`).
        k:
            Optional exact length constraint.

    Returns:
        tuple:
            Tuple containing the sequence elements.

    Raises:
        ValidationError:
            If `value` is not a sequence or its length does not equal
            `k`.
    '''
    if not (
        isinstance(value, collections.abc.Sequence) and
        not isinstance(value, (str, bytes, bytearray))
    ):
        raise ValidationError('Expected a sequence of objects')
    if k is not None and len(value) != k:
        raise ValidationError(f'Length {len(value)} != required {k}')
    return tuple(value)


@typing.overload
def require_numeric_sequence(
    value: object,
    require_type: None = None,
    k: int | None = None,
) -> tuple[int | float, ...]: ...


@typing.overload
def require_numeric_sequence(
    value: object,
    require_type: type[T],
    k: int | None = None,
) -> tuple[T, ...]: ...


def require_numeric_sequence(
    value: object,
    require_type: type[T] | None = None,
    k: int | None = None,
) -> tuple[T, ...] | tuple[int | float, ...]:
    '''
    Require a sequence of numbers with optional length constraint.

    Args:
        value:
            Input object to validate as a sequence of numbers.
        require_type:
            Optional exact numeric type (`int` or `float`) for elements.
        k:
            Optional exact sequence length constraint.

    Returns:
        tuple[T, ...] | tuple[int | float, ...]:
            Tuple of validated numeric elements.

    Raises:
        ValidationError:
            If `value` is not a sequence of length `k`, or any element
            fails numeric validation.
    '''
    _value = require_sequence(value, k=k)
    for v in _value:
        require_numeric(v, require_type)
    return _value


def require_string_sequence(
    value: object,
    k: int | None = None,
) -> tuple[str, ...]:
    '''
    Require a sequence of non-empty strings with optional length.

    Args:
        value:
            Input object to validate as a sequence of strings.
        k:
            Optional exact sequence length constraint.

    Returns:
        tuple[str, ...]:
            Tuple of validated non-empty string elements.

    Raises:
        ValidationError:
            If `value` is not a sequence or fails length constraint `k`.
        TypeError:
            If any element is not a `str`.
        ValueError:
            If any string element is empty.
    '''
    _value = require_sequence(value, k=k)
    for v in _value:
        require_string(v)
    return _value


# files
@typing.overload
def require_file(
    path: str | pathlib.Path | None,
    *,
    missing_ok: typing.Literal[False] = False,
) -> pathlib.Path: ...


@typing.overload
def require_file(
    path: str | pathlib.Path | None,
    *,
    missing_ok: typing.Literal[True],
) -> pathlib.Path | None: ...


@typing.overload
def require_file(
    path: str | pathlib.Path | None,
    *,
    missing_ok: bool = False,
) -> pathlib.Path | None: ...


def require_file(
    path: str | pathlib.Path | None,
    *,
    missing_ok: bool = False,
) -> pathlib.Path | None:
    '''
    Require an existing regular file and return it as a `Path`.

    Args:
        path:
            Candidate file path or `None`.
        missing_ok:
            Whether to return `None` when `path` is `None`.

    Returns:
        pathlib.Path | None:
            Validated file path, or `None` when `path` is `None` and
            `missing_ok` is `True`.

    Raises:
        ValidationError:
            If `path` is `None` (when `missing_ok` is `False`), invalid,
            or not an existing regular file.
    '''
    if path is None:
        if missing_ok:
            return None
        raise ValidationError('Path not provided')

    try:
        _path = pathlib.Path(path)
    except (TypeError, ValueError) as e:
        raise ValidationError(f'Invalid path: {path}') from e

    if not _path.is_file():
        raise ValidationError(f'Not a valid file: {path}')

    return _path


def require_dir(
    path: str | pathlib.Path | None,
    *,
    writable: bool = False,
) -> pathlib.Path:
    '''
    Require an existing directory, optionally probing writability.

    Args:
        path:
            Candidate directory path or `None`.
        writable:
            If `True`, create the directory if needed and verify write
            permissions via a sentinel probe file.

    Returns:
        pathlib.Path:
            Validated directory path.

    Raises:
        ValidationError:
            If `path` is `None`, invalid, not an existing directory, or
            not writable when `writable` is `True`.
    '''
    if path is None:
        raise ValidationError('Path not provided')

    try:
        _path = pathlib.Path(path)
    except (TypeError, ValueError) as e:
        raise ValidationError(f'Invalid path: {path}') from e

    if writable:
        try:
            _path.mkdir(parents=True, exist_ok=True)
            probe_file = _path / '.test'
            probe_file.write_text('probe', encoding='utf-8')
            probe_file.unlink(missing_ok=True)
        except Exception as err:
            raise ValidationError(f'{_path} is not writable: {err}. ') from err
    elif not _path.is_dir():
        raise ValidationError(f'Path {path} is not a directory')

    return _path
