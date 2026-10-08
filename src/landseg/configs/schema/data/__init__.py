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
Top-level namespace for `landseg.configs.schema.data`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataConfig',
    'DataHarmonizationConfig',
    'DataIngestionConfig',
    'DataPreparationConfig',
    'DataSpecificationConfig',
    'WorldGridConfig',
]


# for static check
if typing.TYPE_CHECKING:
    from ._composite import (
        DataConfig,
    )
    from .harmonziation import (
        DataHarmonizationConfig,
    )
    from .ingestion import (
        DataIngestionConfig,
    )
    from .preparation import (
        DataPreparationConfig,
    )
    from .specification import (
        DataSpecificationConfig,
    )
    from .world_grid import (
        WorldGridConfig,
    )


def __getattr__(name: str):

    if name in {'DataConfig'}:
        obj = importlib.import_module('._composite', __package__)
        return getattr(obj, name)

    if name in {'DataHarmonizationConfig'}:
        obj = importlib.import_module('.harmonziation', __package__)
        return getattr(obj, name)

    if name in {'DataIngestionConfig'}:
        obj = importlib.import_module('.ingestion', __package__)
        return getattr(obj, name)

    if name in {'DataPreparationConfig'}:
        obj = importlib.import_module('.preparation', __package__)
        return getattr(obj, name)

    if name in {'DataSpecificationConfig'}:
        obj = importlib.import_module('.specification', __package__)
        return getattr(obj, name)

    if name in {'WorldGridConfig'}:
        obj = importlib.import_module('.world_grid', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
