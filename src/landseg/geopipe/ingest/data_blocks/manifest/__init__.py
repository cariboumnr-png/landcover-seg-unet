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
Top-level namespace for `landseg.geopipe.ingest.data_blocks.manifest`.

Exposes catalog, schema, and manifest lifecycle tools for block-level
geospatial dataset management via lazy module resolution.

Public APIs:
    - ManifestUpdateContext: Dataclass context for manifest update.
    - build_catalog: Builds or updates a dataset-level catalog.
    - build_schema: Creates or updates dataset-level data schema.
    - update_manifest: Updates dataset catalog and schema artifacts.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'ManifestUpdateContext',
    # functions
    'build_catalog',
    'build_schema',
    'update_manifest',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .catalog import build_catalog
    from .lifecycle import ManifestUpdateContext, update_manifest
    from .schema import build_schema


def __getattr__(name: str):
    if name in {'build_catalog'}:
        return getattr(
            importlib.import_module('.catalog', __package__), name
        )

    if name in {'ManifestUpdateContext', 'update_manifest'}:
        return getattr(
            importlib.import_module('.lifecycle', __package__), name
        )

    if name in {'build_schema'}:
        return getattr(
            importlib.import_module('.schema', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

