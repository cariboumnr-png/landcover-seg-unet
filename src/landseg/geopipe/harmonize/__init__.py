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
Top-level namespace for `landseg.geopipe.harmonize`.

Exposes raster harmonization, manifest compilation, taxonomy resolution,
and logging APIs via lazy resolution to keep import order simple and
circular-free.

Public APIs:
    - `HarmonizationLogger`: Logger tracking ETL progress and report JSON.
    - `ProcessedRasters`: Container for processed raster paths.
    - `unify_nodata_mask`: Create a 1-band valid pixel mask across bands.
    - `compile_dataset_manifest`: Read and validate dataset manifest JSON.
    - `harmonize_sources`: Harmonize all compiled raster sources onto grid.
    - `get_available_profiles`: Return registered taxonomy profile names.
    - `validate_specs`: Validate taxonomy specs against knowledge base.
    - `HarmonizationReportSchema`: TypedDict for overall pipeline report.
    - `ProvenanceRecord`: TypedDict for raw raster file provenance.
    - `WorldGridReport`: TypedDict for world grid summary report.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'HarmonizationLogger',
    'ProcessedRasters',
    # functions
    'unify_nodata_mask',
    'compile_dataset_manifest',
    'harmonize_sources',
    'get_available_profiles',
    'validate_specs',
    # typing
    'HarmonizationReportSchema',
    'ProvenanceRecord',
    'WorldGridReport',
]

# for static check
if typing.TYPE_CHECKING:
    from .common import (
        HarmonizationLogger,
        HarmonizationReportSchema,
        ProvenanceRecord,
        WorldGridReport,
    )

    from .manifest import(
        compile_dataset_manifest,
    )

    from .processor import (
        ProcessedRasters,
        harmonize_sources,
    )

    from .rasters import (
        unify_nodata_mask,
    )

    from .taxonomy import(
        get_available_profiles,
        validate_specs,
    )


def __getattr__(name: str):

    if name in {
        'HarmonizationLogger',
        'HarmonizationReportSchema',
        'ProvenanceRecord',
        'WorldGridReport',
    }:
        return getattr(
            importlib.import_module('.common', __package__), name
        )

    if name in {
        'compile_dataset_manifest',
    }:
        return getattr(
            importlib.import_module('.manifest', __package__), name
        )

    if name in {
        'unify_nodata_mask',
    }:
        return getattr(
            importlib.import_module('.rasters', __package__), name
        )

    if name in {
        'ProcessedRasters',
        'harmonize_sources',
    }:
        return getattr(
            importlib.import_module('.processor', __package__), name
        )

    if name in {
        'get_available_profiles',
        'validate_specs',
    }:
        return getattr(
            importlib.import_module('.taxonomy', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
