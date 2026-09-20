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
    - `compile_dataset_manifest`: Read and validate dataset manifest JSON.
    - `DatasetManifestError`: Error raised during manifest compilation.
    - `AllowedCategory`: Type alias for valid raster categories.
    - `FeatureSchemes`: Type alias for feature band scheme mappings.
    - `LabelScheme`: Re-exported TypedDict for label reclassification scheme.
    - `LabelSchemes`: Re-exported alias for label reclassification schemes.
    - `ManifestEntry`: TypedDict defining per-raster configuration shape.
    - `ManifestEntryNormalizer`: Normalize and validate a manifest entry.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # functions
    'compile_dataset_manifest',
    # classes
    'DatasetManifestError',
    # typing
    'ManifestEntry',
]

# for static check
if typing.TYPE_CHECKING:
    from .compiler import (
        compile_dataset_manifest,
        DatasetManifestError,
    )

    from .schema import (
        ManifestEntry,
    )


def __getattr__(name: str):

    if name in {
        'compile_dataset_manifest',
        'DatasetManifestError',
    }:
        return getattr(
            importlib.import_module('.compiler', __package__), name
        )


    if name in {
        'ManifestEntry',
    }:
        return getattr(
            importlib.import_module('.schema', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
