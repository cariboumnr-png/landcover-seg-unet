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
    - `DatasetManifestBuilder`: Fluent builder for dataset manifests.
    - `DatasetManifestError`: Error raised during manifest compilation.
    - `ManifestEntry`: TypedDict defining per-raster configuration.
    - `SensorPreset`: Standard sensor band definitions and presets.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DatasetManifestError',
    'DatasetManifestBuilder',
    # functions
    'normalize_manifest_entry',
    # typing
    'ManifestEntry',
    'SensorPreset',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import (
        DatasetManifestError,
        DatasetManifestBuilder,
    )
    from .normalizer import (
        normalize_manifest_entry,
    )
    from .schema import (
        ManifestEntry,
        SensorPreset,
    )


def __getattr__(name: str):
    if name in {
        'DatasetManifestError',
        'DatasetManifestBuilder',
    }:
        obj = importlib.import_module('.builder', __package__)
        return getattr(obj, name)

    if name in {
        'normalize_manifest_entry',
    }:
        obj = importlib.import_module('.normalizer', __package__)
        return getattr(obj, name)

    if name in {
        'ManifestEntry',
        'SensorPreset',
    }:
        obj = importlib.import_module('.schema', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
