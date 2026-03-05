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

'''
Type guards and TypedDict definitions for validating composite loss
configuration. Ensures focal and dice loss blocks follow expected
schemas before constructing CompositeLoss modules.
'''

# standard imports
from __future__ import annotations
import typing

# ---------------------------------Public Type---------------------------------
class LossTypes(typing.TypedDict, total=False):
    '''Typed composite loss config supporting focal and dice entries.'''
    focal: _FocalConfig
    dice: _DiceConfig

# --------------------------------private  type--------------------------------
class _FocalConfig(typing.TypedDict):
    '''TypedDict schema for focal loss parameters.'''
    weight: float
    alpha: list[float] | None
    gamma: float
    reduction: str

class _DiceConfig(typing.TypedDict):
    '''TypedDict schema for dice loss parameters.'''
    weight: float
    smooth: float

# -------------------------------Public Function-------------------------------
def is_loss_types(cfg: dict) -> typing.TypeGuard[LossTypes]:
    '''Validate loss config against supported TypedDict schemas.'''

    # current types: focal, dice
    has_focal = False
    has_dice = False
    # check if each exists
    if _is_focal(cfg.get('focal')):
        has_focal = True
    elif _is_dice(cfg.get('dice')):
        has_dice = True
    # needs to contain at least one of the types
    return has_focal or has_dice

def _is_focal(d: dict | None) -> bool:
    '''Check whether dict matches focal-loss parameter schema.'''
    if d is None:
        return False
    return (
        isinstance(d.get('weight'), float) and
        _is_alpha(d.get('alpha')) and
        isinstance(d.get('gamma'), float) and
        isinstance(d.get('reduction'), str)
    )

def _is_dice(d: dict | None) -> bool:
    '''Check whether dict matches dice-loss parameter schema.'''
    if d is None:
        return False
    return (
        isinstance(d.get('weight'), float) and
        isinstance(d.get('smooth'), float)
    )

def _is_alpha(a: list | None) -> bool:
    '''Validate a as None or a list of float values.'''
    # can be a list of float or just None
    return (
        a is None or
        (isinstance(a, list) and all(isinstance(x, float) for x in a))
    )
