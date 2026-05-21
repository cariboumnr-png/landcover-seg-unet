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
Top-level namespace for `landseg.models.core`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DomainContextRouter',
    'DomainTargetPayload',
    'DomainProjectionConfig',
    'DomainTargetConfig',
    'HeadManager',
    'NumericSafety',
    'ConcatAdapter',
    'FilmConditioner',
    # functions
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .conditioner import ConcatAdapter, FilmConditioner
    from .config import DomainProjectionConfig, DomainTargetConfig
    from .domains import DomainContextRouter, DomainTargetPayload
    from .heads import HeadManager
    from .safety import NumericSafety

def __getattr__(name: str):

    if name in {'ConcatAdapter', 'FilmConditioner'}:
        return getattr(importlib.import_module('.conditioner', __package__), name)

    if name in {'DomainProjectionConfig', 'DomainTargetConfig'}:
        return getattr(importlib.import_module('.config', __package__), name)

    if name in {'DomainContextRouter', 'DomainTargetPayload'}:
        return getattr(importlib.import_module('.domains', __package__), name)

    if name in {'HeadManager'}:
        return getattr(importlib.import_module('.heads', __package__), name)

    if name in {'NumericSafety'}:
        return getattr(importlib.import_module('.safety', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
