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
Top-level namespace for `landseg.geopipe`.

Coordinates raw geospatial raster harmonization, domain tile mapping,
data block construction, stage execution, and runtime dataset
specification creation through structured pipelines and lazy resolution.

Public APIs:
    - `DatasetManifestBuilder`: Fluent builder for dataset manifests.
    - `SensorPreset`: Standard sensor band definitions and presets.
    - `build_dataspec`: Assembles runtime DataSpecs from artifacts.
    - `build_overfit_dataspec`: Build single-block DataSpecs for overfit.
    - `build_vrt_mosaic`: Merge multiple raster tiles into a mosaic VRT.
    - `resolve_pending_ingestion_batches`: Resolve pending ingest batches.
    - `run_data_harmonization_stage`: Execute data-harmonize stage.
    - `run_data_ingestion_stage`: Execute data-ingest stage.
    - `run_data_preparation_stage`: Execute data-prepare stage.
    - `run_world_grid_stage`: Execute world-grid generation stage.
    - `validate_feature_specs`: Validate requested feature specs.
    - `validate_grid_reference`: Validate a persisted world grid reference.
    - `validate_grid_specs`: Validate world grid definition inputs.
    - `validate_manifest_contract`: Validate input dataset manifest entries.
    - `validate_pool_compatibility`: Validate grid/pool compatibility.
    - `validate_prepare_semantics`: Validate preparation feature & head specs.
    - `validate_raster_alignment`: Validate raster/grid alignment for ingest.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'DatasetManifestBuilder',
    'SensorPreset',
    'build_dataspec',
    'build_overfit_dataspec',
    'build_vrt_mosaic',
    'resolve_pending_ingestion_batches',
    'run_data_harmonization_stage',
    'run_data_ingestion_stage',
    'run_data_preparation_stage',
    'run_world_grid_stage',
    'validate_feature_specs',
    'validate_grid_reference',
    'validate_grid_specs',
    'validate_manifest_contract',
    'validate_pool_compatibility',
    'validate_prepare_semantics',
    'validate_raster_alignment',
]


# for static check
if typing.TYPE_CHECKING:
    from .diagnostic import build_overfit_dataspec
    from .factory import build_dataspec
    from .harmonize.manifest import (
        DatasetManifestBuilder,
        SensorPreset,
    )
    from .ingest import resolve_pending_ingestion_batches
    from .rasters import build_vrt_mosaic
    from .stages import (
        run_data_harmonization_stage,
        run_data_ingestion_stage,
        run_data_preparation_stage,
        run_world_grid_stage,
    )
    from .validation import (
        validate_feature_specs,
        validate_grid_reference,
        validate_grid_specs,
        validate_manifest_contract,
        validate_pool_compatibility,
        validate_prepare_semantics,
        validate_raster_alignment,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.diagnostic': ['build_overfit_dataspec'],
    '.factory': ['build_dataspec'],
    '.harmonize.manifest': [
        'DatasetManifestBuilder',
        'SensorPreset',
    ],
    '.ingest': ['resolve_pending_ingestion_batches'],
    '.rasters': ['build_vrt_mosaic'],
    '.stages': [
        'run_data_harmonization_stage',
        'run_data_ingestion_stage',
        'run_data_preparation_stage',
        'run_world_grid_stage',
    ],
    '.validation': [
        'validate_feature_specs',
        'validate_grid_reference',
        'validate_grid_specs',
        'validate_manifest_contract',
        'validate_pool_compatibility',
        'validate_prepare_semantics',
        'validate_raster_alignment',
    ],
})
