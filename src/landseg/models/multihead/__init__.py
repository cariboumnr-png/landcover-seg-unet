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
Top-level namespace for `landseg.models.multihead`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BackboneConfig',
    'BaseMultiheadModel',
    'ConcatConfig',
    'ConditioningConfig',
    'FilmConfig',
     'ModelConfig',
    'MultiHeadUNet',
    # functions
    'get_concat',
    'get_film'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .base import BaseMultiheadModel
    from .concat import get_concat
    from .config import BackboneConfig, ConcatConfig, ConditioningConfig, FilmConfig, ModelConfig
    from .film import get_film
    from .frame import MultiHeadUNet

def __getattr__(name: str):

    if name in ['BaseMultiheadModel']:
        return getattr(importlib.import_module('.base', __package__), name)
    if name in ['get_concat']:
        return getattr(importlib.import_module('.concat', __package__), name)
    if name in ['BackboneConfig', 'ConcatConfig', 'ConditioningConfig',
                'FilmConfig', 'ModelConfig']:
        return getattr(importlib.import_module('.config', __package__), name)
    if name in ['get_film']:
        return getattr(importlib.import_module('.film', __package__), name)
    if name in ['MultiHeadUNet']:
        return getattr(importlib.import_module('.frame', __package__), name)

    raise AttributeError(name)
