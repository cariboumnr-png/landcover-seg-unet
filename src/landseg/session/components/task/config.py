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
# pylint: disable=too-few-public-methods

'''
Common types for task module.
'''

# standard typing
from __future__ import annotations
import typing

# ---------------------------------Public Type---------------------------------
class TaskConfig(typing.Protocol):
    '''doc'''
    @property
    def alpha_fn(self) -> str: ...
    @property
    def en_beta(self) -> float: ...
    @property
    def excluded_cls(self) -> dict[str, list[int]] | None: ...
    @property
    def types(self) -> _LossTypes: ...

# --------------------------------private  type--------------------------------
class _LossTypes(typing.Protocol):
    @property
    def focal(self) -> _FocalLoss: ...
    @property
    def dice(self) -> _DiceLoss: ...
    @property
    def spectral(self) -> _SpectralLoss: ...
    @property
    def tv(self) -> _TotalVariationLoss: ...

class _FocalLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
    @property
    def gamma(self) -> float: ...
    @property
    def reduction(self) -> str: ...

class _DiceLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
    @property
    def smooth(self) -> float: ...

class _SpectralLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def neighbour(self) -> int: ...

class _TotalVariationLoss(typing.Protocol):
    @property
    def weight(self) -> float: ...
