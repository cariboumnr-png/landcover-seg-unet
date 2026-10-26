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
import pathlib
import typing
# local imports
import landseg.utils.validator as val

# ----- typing
V = typing.TypeVar('V')


# ----- typing aliases
ConfigValidationError = val.ValidationError


# ----- public classes
class BaseConfigSection:
    '''Base configuration class providing shared validation primitives.'''

    @property
    def as_dict(self) -> dict[str, typing.Any]:
        '''Return dictionary representation of the dataclass section.'''
        return dataclasses.asdict(typing.cast(typing.Any, self))

    def validate(self) -> None:
        '''Validate section integrity. Default is a no-op.'''
        return None

    @staticmethod
    def require_types(
        value: object,
        name: str,
        required_type: type[V] | tuple[type[V], ...]
    ) -> V:
        '''Require a type or a tuple of types.'''
        return val.require_types(value, name, required_type)

    @staticmethod
    def require_str(
        value: object,
        name: str,
    ) -> str:
        '''Require a string from [allowed].'''
        return val.require_types(value, name, str)

    @staticmethod
    def require_allowed_str(
        value: object,
        name: str,
        allowed_str: typing.Sequence[str]
    ) -> str:
        '''Require a string from [allowed].'''
        return val.require_value_from(value, name, allowed_str, str)

    @staticmethod
    def require_int(
        value: object,
        name: str,
    ) -> int:
        '''Require an integer from [allowed].'''
        return val.require_numeric(value, name, int)

    @staticmethod
    def require_allowed_int(
        value: object,
        name: str,
        allowed_int: typing.Sequence[int]
    ) -> int:
        '''Require an integer from [allowed].'''
        return val.require_value_from(value, name, allowed_int, int)

    @staticmethod
    def require_int_min(
        value: object,
        name: str,
        value_min: int
    ) -> int:
        '''Require an integer >= minimal'''
        return val.require_numeric_min(value, name, value_min, int)

    @staticmethod
    def require_int_range(
        value: object,
        name: str,
        value_range: tuple[int, int],
    ) -> int:
        '''Require an integer within range.'''
        return val.require_numeric_range(value, name, value_range, int)

    @staticmethod
    def require_float_min(
        value: object,
        name: str,
        value_min: float
    ) -> float:
        '''Require a float >= minimal'''
        return val.require_numeric_min(value, name, value_min, float)

    @staticmethod
    def require_float_max(
        value: object,
        name: str,
        value_max: float
    ) -> float:
        '''Require a float <= maximum'''
        return val.require_numeric_max(value, name, value_max, float)

    @staticmethod
    def require_float_range(
        value: object,
        name: str,
        value_range: tuple[float, float],
    ) -> float:
        '''Require a float within range.'''
        return val.require_numeric_range(value, name, value_range, float)

    @staticmethod
    def require_sequence(
        value: object,
        name: str,
        k: int | None = None
    ) -> tuple:
        '''Return a sequence with expected type and length as a tuple.'''
        return val.require_sequence(value, name, k=k)

    @staticmethod
    def require_file(
        path: str | None,
        file_name: str
    ) -> pathlib.Path | None:
        '''Raise if filepath is provided but does not exist.'''
        return val.require_file(path, file_name, missing_ok=True)
