# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Diagnostic probe library for pre-flight readiness checks.

Exposes specialized probes for hardware, filesystem storage, and
lineage validation.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'probe_hardware',
    'probe_lineage',
    'probe_storage',
]


# for static check
if typing.TYPE_CHECKING:
    from .hardware import (
        probe_hardware,
    )
    from .lineage import (
        probe_lineage,
    )
    from .storage import (
        probe_storage,
    )


def __getattr__(name: str):
    if name in {'probe_hardware'}:
        obj = importlib.import_module('.hardware', __package__)
        return getattr(obj, name)

    if name in {'probe_lineage'}:
        obj = importlib.import_module('.lineage', __package__)
        return getattr(obj, name)

    if name in {'probe_storage'}:
        obj = importlib.import_module('.storage', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
