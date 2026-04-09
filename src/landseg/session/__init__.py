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
Top-level namespace for `landseg.trainer`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'MultiHeadTrainer',
    'Phase',
    'Runner',
    # functions
    'build_trainer_components',
    # types
    'TrainerConfigShape',
]

# for static check
if typing.TYPE_CHECKING:
    from .common import TrainerConfigShape
    from .components import build_trainer_components
    from .engine import MultiHeadTrainer
    from .runner import Phase, Runner

def __getattr__(name: str):

    if name in {'TrainerConfigShape'}:
        return getattr(importlib.import_module('.common', __package__), name)

    if name in {'Phase', 'HeadsConifg', 'LogitAdjustScheme', 'Runner'}:
        return getattr(importlib.import_module('.runner', __package__), name)

    if name in {'MultiHeadTrainer'}:
        return getattr(importlib.import_module('.engine', __package__), name)

    if name in {'build_trainer_components',}:
        return getattr(importlib.import_module('.components', __package__), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
