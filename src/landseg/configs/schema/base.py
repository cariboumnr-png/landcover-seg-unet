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
import typing
# local imports
import landseg.utils.validator as validator


# ----- typing aliases
ConfigValidationError = validator.ValidationError


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

    val = validator
