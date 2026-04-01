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
Top-level namespace for `landseg.geopipe.artifacts`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'LifecyclePolicy',
    # functions
    'load_json_hash',
    'load_pickle_hash',
    'write_json_hash',
    'write_pickle_hash',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .json_io import load_json_hash, write_json_hash
    from .pickle_io import load_pickle_hash, write_pickle_hash
    from .policy import LifecyclePolicy

def __getattr__(name: str):

    if name in {'load_json_hash', 'write_json_hash'}:
        return getattr(importlib.import_module('.json_io', __package__), name)

    if name in {'load_pickle_hash', 'write_pickle_hash'}:
        return getattr(importlib.import_module('.pickle_io', __package__), name)

    if name in {'LifecyclePolicy'}:
        return getattr(importlib.import_module('.policy', __package__), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
