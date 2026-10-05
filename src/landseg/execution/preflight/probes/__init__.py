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

Exposes specialized probes for hardware, filesystem storage, spatial
contracts, run ledgers, model specifications, and optimization studies.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'probe_hardware',
    'probe_ledger',
    'probe_lineage',
    'probe_model',
    'probe_spatial',
    'probe_storage',
    'probe_study',
]


# for static check
if typing.TYPE_CHECKING:
    from .hardware import (
        probe_hardware,
    )
    from .ledger import (
        probe_ledger,
    )
    from .lineage import (
        probe_lineage,
    )
    from .model import (
        probe_model,
    )
    from .spatial import (
        probe_spatial,
    )
    from .storage import (
        probe_storage,
    )
    from .study import (
        probe_study,
    )


def __getattr__(name: str):
    if name in {'probe_hardware'}:
        obj = importlib.import_module('.hardware', __package__)
        return getattr(obj, name)

    if name in {'probe_ledger'}:
        obj = importlib.import_module('.ledger', __package__)
        return getattr(obj, name)

    if name in {'probe_lineage'}:
        obj = importlib.import_module('.lineage', __package__)
        return getattr(obj, name)

    if name in {'probe_model'}:
        obj = importlib.import_module('.model', __package__)
        return getattr(obj, name)

    if name in {'probe_spatial'}:
        obj = importlib.import_module('.spatial', __package__)
        return getattr(obj, name)

    if name in {'probe_storage'}:
        obj = importlib.import_module('.storage', __package__)
        return getattr(obj, name)

    if name in {'probe_study'}:
        obj = importlib.import_module('.study', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
