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
Top-level namespace for `landseg.geopipe.harmonize.manifest`.

Exposes dataset manifest schemas, compilers, and normalizers.

Public APIs:
    - `DatasetManifestError`: Error raised during manifest compilation.
    - `compile_dataset_manifest`: Read and validate dataset manifest.
    - `ManifestEntry`: TypedDict defining per-raster configuration.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DatasetManifestError',
    # functions
    'compile_dataset_manifest',
    # typing
    'ManifestEntry',
]


# for static check
if typing.TYPE_CHECKING:
    from .compiler import (
        DatasetManifestError,
        compile_dataset_manifest,
    )
    from .schema import (
        ManifestEntry,
    )


def __getattr__(name: str):
    if name in {
        'DatasetManifestError',
        'compile_dataset_manifest',
    }:
        obj = importlib.import_module('.compiler', __package__)
        return getattr(obj, name)

    if name in {'ManifestEntry'}:
        obj = importlib.import_module('.schema', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
