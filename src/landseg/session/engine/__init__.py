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
Top-level namespace for `landseg.session.engine`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BatchExecutionEngine',
    'EngineBase',
    'MultiHeadEvaluator',
    'MultiHeadTrainer',
    # functions
    'init_state'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .base import EngineBase
    from .core import BatchExecutionEngine, init_state
    from .evaluator import MultiHeadEvaluator
    from .trainer import MultiHeadTrainer

def __getattr__(name: str):

    if name in {'EngineBase'}:
        return getattr(importlib.import_module('.base', __package__), name)

    if name in {'BatchExecutionEngine', 'init_state'}:
        return getattr(importlib.import_module('.core', __package__), name)

    if name in {'MultiHeadEvaluator'}:
        return getattr(importlib.import_module('.evaluator', __package__), name)

    if name in {'MultiHeadTrainer'}:
        return getattr(importlib.import_module('.trainer', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
