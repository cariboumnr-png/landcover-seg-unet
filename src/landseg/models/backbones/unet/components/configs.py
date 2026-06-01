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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''Backbone components configurations.'''

# standard imports
from __future__ import annotations
import typing

class BottleneckConfig(typing.Protocol):
    @property
    def variant(self) -> str: ...
    @property
    def num_conv_blocks(self) -> int | None: ...
    @property
    def conv_params(self) -> ConvolutionParameters | None: ...
    @property
    def num_transformer_blocks(self) -> int | None: ...
    @property
    def transformer_params(self) -> TransformerParameters | None: ...

class ConvolutionParameters(typing.Protocol):
    @property
    def norm(self) -> str | None: ...
    @property
    def gn_groups(self) -> int | None: ...
    @property
    def p_drop(self) -> float: ...

class TransformerParameters(typing.Protocol):
    @property
    def num_heads(self) -> int: ...
    @property
    def mlp_ratio(self) -> float: ...
    @property
    def dropout(self) -> float: ...
    @property
    def attn_dropout(self) -> float: ...
