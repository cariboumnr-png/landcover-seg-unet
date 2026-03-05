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
Top-level namespace for `landseg.training.loss`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'CompositeLoss',
    'DiceLoss',
    'FocalLoss',
    'PrimitiveLoss',
    # functions
    'build_headlosses',
    'is_loss_types',
]
# for static check
if typing.TYPE_CHECKING:
    from .base import PrimitiveLoss
    from .composite import CompositeLoss
    from .primitives import DiceLoss, FocalLoss
    from .factory import build_headlosses
    from .validator import is_loss_types


def __getattr__(name: str):

    if name == 'PrimitiveLoss':
        return importlib.import_module('.base', __package__).PrimitiveLoss
    if name == 'CompositeLoss':
        return importlib.import_module('.composite', __package__).CompositeLoss
    if name in ('DiceLoss', 'FocalLoss'):
        return getattr(importlib.import_module('.primitives', __package__), name)
    if name == 'build_headlosses':
        return importlib.import_module('.factory', __package__).build_headlosses
    if name == 'is_loss_types':
        return importlib.import_module('.validator', __package__).is_loss_types

    raise AttributeError(name)
