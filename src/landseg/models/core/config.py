# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-function-docstring

'''Multihead model typed configuration.'''

from __future__ import annotations
# standard imports
import typing

class DomainTargetConfig(typing.Protocol):
    '''Typed container for configuring concatenation adapter.'''
    @property
    def use_ids(self) -> bool: ...
    @property
    def use_vec(self) -> bool: ...
    @property
    def projection(self) -> DomainProjectionConfig: ...

class DomainProjectionConfig(typing.Protocol):
    '''Projection configuration for one domain target.'''
    @property
    def out_dim(self) -> int: ...
    @property
    def use_mlp(self) -> bool: ...
    @property
    def hidden_dim(self) -> int | None: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def dropout(self) -> float: ...
    @property
    def activation(self) -> str: ...
