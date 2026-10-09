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
Top-level namespace for `landseg.geopipe.prepare.dataset`.

Exposes in-memory dataset view containers and builders via lazy resolution
to keep import order simple and circular-free.

Public APIs:
    - `DatasetView`: unified immutable dataset preparation view.
    - `DatasetViewConfig`: configuration parameters for dataset view.
    - `build_dataset_view`: construct full preparation view.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataBlocksManifestInputs',
    'DatasetView',
    'DatasetViewConfig',
    # functions
    'build_dataset_view',
]


# for static check
if typing.TYPE_CHECKING:
    from .normalizer import (
        DataBlocksManifestInputs
    )
    from .view import (
        DatasetView,
        DatasetViewConfig,
        build_dataset_view,
    )


def __getattr__(name: str):
    if name in {
        'DataBlocksManifestInputs',
    }:
        obj = importlib.import_module('.normalizer', __package__)
        return getattr(obj, name)

    if name in {
        'DatasetView',
        'DatasetViewConfig',
        'build_dataset_view',
    }:
        obj = importlib.import_module('.view', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
