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

'''Base class for a configuration section'''

# standard imports
import dataclasses
import os
import typing

T = typing.TypeVar('T')

NumericRange: typing.TypeAlias = tuple[int | float | None, int | float | None]


class ConfigValidationError(Exception):
    '''Configuration Error'''
    def __init__(self, message='Error validation configs'):
        super().__init__(message)


class BaseConfigSection:
    '''Base configuration class providing shared validation primitives.'''

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return dictionary representation of the dataclass section.'''
        return dataclasses.asdict(typing.cast(typing.Any, self))

    def validate(self) -> None:
        '''Validate section integrity. Default is a no-op.'''
        return None

    def require_attr_type_range(
        self,
        attr_name: str,
        require_types: type[T] | tuple[type[T], ...],
        require_range: NumericRange | tuple[str] | list[str] | None = None,
    ) -> None:
        '''Raise TypeError if input is not of required type'''
        # handle missing attribute (unlikely since only called internally)
        try:
            value = getattr(self, attr_name)
        except AttributeError as e:
            raise ConfigValidationError(
                f'Unable to retrive attribute {attr_name} from'
                f'{type(self).__name__}'
            ) from e

        try:
            self.require_type_range(value, attr_name, require_types, require_range)
        except ValueError as e:
            raise ConfigValidationError(
                f'Failed to validate attribute {attr_name} value'
            ) from e

    @staticmethod
    def require_type_range(
        value: object,
        value_name: str,
        require_types: type[T] | tuple[type[T], ...],
        require_range: NumericRange | tuple[str] | list[str] | None = None,
    ) -> None:
        '''Raise ValueError if input is not ∈ required type | range.'''

        def _is_numeric_range(v) -> NumericRange:
            if not isinstance(v, tuple) and len(v) == 2:
                raise ValueError('Numeric range not a tuple of two elements')
            a, b = v
            if not (
                (a is None or isinstance(a, (int | float))) and
                (b is None or isinstance(b, (int | float)))
            ):
                raise ValueError('Numeric range can only have int/float/None')
            if a is None and b is None:
                raise ValueError('Numeric range cannot have both ends as None')
            return v

        def _is_literal_range(v) -> list[str]:
            if (
                not isinstance(v, (tuple, list)) and
                all(isinstance(vv, str) for vv in v)
            ):
                raise ValueError('Literal range must be a list/tuple of str')
            return [vv.lower for vv in v]

        if value is None:
            return # skip if value is None type

        if not isinstance(value, require_types):
            if isinstance(require_types, tuple):
                required = ', '.join(t.__name__ for t in require_types)
            else:
                required = require_types.__name__

            raise ValueError(
                f'Expected type: {required} for {value_name}, '
                f'got {value} with type: {type(value)} instead'
            )

        if require_range is None:
            return # skip if range is not to be checked

        if isinstance(value, (int, float)):
            vmin, vmax = _is_numeric_range(require_range)
            if (
                (vmin is not None and value < vmin) or
                (vmax is not None and value > vmax)
            ):
                raise ValueError(
                    f'Value [{value_name}] must be within [{vmin}, {vmax}], '
                    f'got: {value} instead'
                )

        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f'Value [{value_name}] must be non-empty')

            allowed = _is_literal_range(require_range)
            if value.lower() not in allowed:
                raise ValueError(
                    f'Value [{value_name}] must be in the following: {allowed} '
                    f'got: {value} instead'
                )

    @staticmethod
    def require_file(
        file_path: str | None,
        file_name: str
    ) -> None:
        '''Raise FileNotFoundError if file does not exist'''
        if file_path and not BaseConfigSection.file_exists(file_path):
            raise ConfigValidationError(
                f'File [{file_name}] is not file or does not exist at: '
                f'{file_path}'
            )

    @staticmethod
    def file_exists(path: str) -> bool:
        '''If file exists, return True'''
        return os.path.isfile(path) and os.path.exists(path)
