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
Top-level namespace for `landseg.session.contracts`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # typing
    'DatasetItem',
    'PhaseLike',
    'SessionObserverLike',
    'TensorDict',
]


# for static check
if typing.TYPE_CHECKING:
    from .data import(
        TensorDict,
        DatasetItem
    )

    from .observer import (
        SessionObserverLike,
    )

    from .phases import (
        PhaseLike,
    )


def __getattr__(name: str):
    if name in {'DatasetItem', 'TensorDict'}:
        obj = importlib.import_module('.data', __package__)
        return getattr(obj, name)

    if name in {'SessionObserverLike'}:
        obj = importlib.import_module('.observer', __package__)
        return getattr(obj, name)

    if name in {'PhaseLike'}:
        obj = importlib.import_module('.phases', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
