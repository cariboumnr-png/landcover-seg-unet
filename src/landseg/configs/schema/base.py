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
    def file_exists(path: str) -> bool:
        '''If file exists, return True'''
        return os.path.isfile(path) and os.path.exists(path)

    @staticmethod
    def must_exist(path: str | None, tag: str) -> None:
        '''Raise FileNotFoundError if file does not exist'''
        if path and not BaseConfigSection.file_exists(path):
            raise FileNotFoundError(f'File [{tag}] is invalid: {path}')

    @staticmethod
    def must_within(
        value: typing.Any,
        tag: str,
        mmin: int | float | None = None,
        mmax: int | float | None = None,
    ) -> None:
        '''Raise ValueError if value is outside the specified range'''
        if not isinstance(value, (int, float)):
            return
        rr = f'[{mmin}, {mmax}]'
        if (
            (mmin is not None and value < mmin) or
            (mmax is not None and value > mmax)
        ):
            raise ValueError(f'Value [{tag}] must be within {rr}, got: {value}')
