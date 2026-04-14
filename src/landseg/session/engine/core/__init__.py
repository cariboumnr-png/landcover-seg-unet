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
Top-level namespace for `landseg.session.engine.core`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BatchExecutionEngine',
    'RuntimeState',
    # functions
    'init_state',
    'multihead_loss',
]

# for static check
if typing.TYPE_CHECKING:
    from .execution import BatchExecutionEngine
    from .loss import multihead_loss
    from .state import RuntimeState, init_state

def __getattr__(name: str):

    if name in {'BatchExecutionEngine'}:
        return getattr(importlib.import_module('.execution', __package__), name)

    if name in {'multihead_loss'}:
        return getattr(importlib.import_module('.loss', __package__), name)

    if name in {'RuntimeState', 'init_state'}:
        return getattr(importlib.import_module('.state', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
