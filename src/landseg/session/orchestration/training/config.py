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

'''Training-focused phase protocol.'''

# standard imports
from __future__ import annotations
import typing

class HeadsConfigLike(typing.Protocol):
    '''Shape of the heads configuration container.'''
    @property
    def active_heads(self) -> list[str]: ...
    @property
    def frozen_heads(self) -> list[str] | None: ...
    @property
    def excluded_cls(self) -> dict[str, list[int]] | None: ...

class PhaseLike(typing.Protocol):
    '''Training phase container'''
    @property
    def name(self)-> str: ...
    @property
    def num_epochs(self)-> int: ...
    @property
    def lr_scale(self)-> float: ...
    @property
    def heads(self)-> HeadsConfigLike: ...
