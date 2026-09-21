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
Top-level namespace for `landseg.session.common`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'SessionLogger',
    # typing
    'PhaseLike',
    'SessionObserverLike',
]


# for static check
if typing.TYPE_CHECKING:
    from .events import (
        SessionObserverLike,
    )
    from .logger import (
        SessionLogger,
    )
    from .phases import (
        PhaseLike,
    )


def __getattr__(name: str):
    if name in {'SessionObserverLike'}:
        obj = importlib.import_module('.events', __package__)
        return getattr(obj, name)

    if name in {'SessionLogger'}:
        obj = importlib.import_module('.logger', __package__)
        return getattr(obj, name)

    if name in {'PhaseLike'}:
        obj = importlib.import_module('.orchestration', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
