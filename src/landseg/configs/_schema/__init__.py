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

# pylint: disable=too-many-return-statements

'''
Top-level namespace for `landseg.configs`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataFoundation',
    'DataTransform',
    'DataSpecs',
    'ModelsConfig',
    'SessionConfig',
    'StudyConfig',
    'PipelineConfig',
    # functions
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .foundation import DataFoundation
    from .transform import DataTransform
    from .dataspecs import DataSpecs
    from .models import ModelsConfig
    from .session import SessionConfig
    from .study import StudyConfig
    from .pipeline import PipelineConfig

def __getattr__(name: str):

    if name in {'DataFoundation'}:
        return getattr(importlib.import_module('.foundation', __package__), name)

    if name in {'DataTransform'}:
        return getattr(importlib.import_module('.transform', __package__), name)

    if name in {'DataSpecs'}:
        return getattr(importlib.import_module('.dataspecs', __package__), name)

    if name in {'ModelsConfig'}:
        return getattr(importlib.import_module('.models', __package__), name)

    if name in {'SessionConfig'}:
        return getattr(importlib.import_module('.session', __package__), name)

    if name in {'StudyConfig'}:
        return getattr(importlib.import_module('.study', __package__), name)

    if name in {'PipelineConfig'}:
        return getattr(importlib.import_module('.pipeline', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
