# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Top-level namespace for `landseg.geopipe.prepare.materialize`.

Exposes block materialization and image normalization pipeline runners
via lazy resolution.

Public APIs:
    - run_materialize_blocks: orchestrate stats aggregation and block
      materialization.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'run_materialize_blocks',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .runner import run_materialize_blocks


def __getattr__(name: str):

    if name in {'run_materialize_blocks'}:
        return getattr(importlib.import_module('.runner', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
