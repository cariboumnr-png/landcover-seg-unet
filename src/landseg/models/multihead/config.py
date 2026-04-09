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

class DataSpecsConfig(typing.Protocol):
    '''General configuration'''
    @property
    def in_ch(self) -> int: ...
    @property
    def logit_adjust(self) -> dict[str, list[float]]: ...
    @property
    def heads_w_counts(self) -> dict[str, list[int]]: ...
    @property
    def domain_ids_num(self) -> int: ...     # id categories
    @property
    def domain_vec_dim(self) -> int: ...     # vector dims

class BackboneConfig(typing.Protocol):
    '''Typed container for model backbone configuration.'''
    @property
    def body(self) -> str: ...
    @property
    def base_ch(self) -> int: ...
    @property
    def conv_params(self) -> dict[str, typing.Any]:...

class ConditioningConfig(typing.Protocol):
    '''Typed container for model conditioning configuration.'''
    @property
    def mode(self) -> str | None: ...
    @property
    def concat(self) -> _ConcatConfig: ...
    @property
    def film(self) -> _FilmConfig: ...

class _ConcatConfig(typing.Protocol):
    '''Typed container for configuring concatenation adapter.'''
    @property
    def out_dim(self) -> int: ...
    @property
    def use_ids(self) -> bool: ...
    @property
    def use_vec(self) -> bool: ...
    @property
    def use_mlp(self) -> bool: ...

class _FilmConfig(typing.Protocol):
    '''Typed container for configuring FiLM conditioner.'''
    @property
    def embed_dim(self) -> int:...
    @property
    def use_ids(self) -> bool: ...
    @property
    def use_vec(self) -> bool: ...
    @property
    def hidden(self) -> int: ...
