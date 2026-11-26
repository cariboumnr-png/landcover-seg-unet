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
Multi-head model typed configuration contracts.

Defines `Protocol` and `TypedDict` interfaces for domain conditioning
configuration, enabling static type checking without runtime dependency
on configuration frameworks such as Hydra.

Public APIs:
    - `DomainConditionerAdapterConfig`: TypedDict for adapter options.
    - `DomainProjectionConfig`: TypedDict for vector projection options.
    - `DomainTargetConfig`: Protocol for domain target configuration.
'''

# standard imports
import typing


class DomainProjectionConfig(typing.TypedDict):
    '''Typed dictionary for domain vector projection configuration.'''
    use_mlp: typing.NotRequired[bool]
    hidden_dim: typing.NotRequired[int | None]
    num_hidden_layers: typing.NotRequired[int]
    dropout: typing.NotRequired[float]
    activation: typing.NotRequired[str]


class DomainConditionerAdapterConfig(typing.TypedDict):
    '''Typed dictionary for domain conditioner adapter options.'''
    hidden_dim: typing.NotRequired[int] # currently for FiLM


# pylint: disable=missing-function-docstring
class DomainTargetConfig(typing.Protocol):
    '''Protocol for domain target conditioning configuration.'''
    @property
    def name(self) -> str: ...
    @property
    def use_ids(self) -> bool: ...
    @property
    def use_vec(self) -> bool: ...
    @property
    def ids_embd_dims(self) -> int: ...
    @property
    def vec_proj_dims(self) -> int: ...
    @property
    def vec_proj_config(self) -> DomainProjectionConfig: ...
    @property
    def conditioner_config(self) -> DomainConditionerAdapterConfig: ...
# pylint: enable=missing-function-docstring
