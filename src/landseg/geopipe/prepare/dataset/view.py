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
Dataset view orchestration utilities for preparation.

Provides a unified builder coordinating catalog reading, feature
channel selection, and target head hierarchy resolution into a single
self-contained in-memory dataset view.

Public APIs:
    - `DatasetView`: unified immutable dataset preparation view.
    - `DatasetViewConfig`: configuration parameters for dataset view.
    - `build_dataset_view`: construct full preparation view.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# third-party imports
import rasterio.transform
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.dataset.normalizer as normalizer
import landseg.geopipe.prepare.dataset.semantics as semantics


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class DatasetViewConfig:
    '''Configuration parameters for dataset view compilation.'''
    head_pxs_thres: dict[str, float] = dataclasses.field(default_factory=dict)
    focal_head: str | None = None
    non_overlapping_test_grid: bool = True
    features: typing.Mapping[str, str] | list[str] | None = None
    targets: typing.Mapping[str, str | geo_core.LabelScheme] | None = None


@dataclasses.dataclass(frozen=True)
class DatasetView:
    '''Unified dataset view for partitioning and statistics.'''
    focal_head: str
    crs: str
    transform: rasterio.transform.Affine
    manifest: normalizer.NormalizedDataBlocksView
    features: semantics.FeatureSelection
    targets: semantics.TargetHeadsContext


# ----- public functions
def build_dataset_view(
    inputs: normalizer.DataBlocksManifestInputs,
    config: DatasetViewConfig,
    *,
    canvas_crs: str,
    canvas_transform: rasterio.transform.Affine,
) -> DatasetView:
    '''
    Build complete dataset view from catalog, schema, and parameters.

    Orchestrates catalog parsing, feature channel resolution, and
    multi-head target hierarchy construction into a unified view.
    Derives per-class pixel counts for all target heads in memory
    without reading block files or raster data from disk.

    Args:
        catalog_fpath:
            path to canonical blocks catalog JSON.
        schema:
            ingested dataset schema dictionary.
        parameters:
            optional configuration parameters for filtering, features,
            and targets.
        canvas_crs:
            optional pre-resolved CRS string of the dataset canvas.
        canvas_transform:
            optional pre-resolved Affine transform of the canvas.

    Returns:
        DatasetView:
            unified preparation view combining catalog, features,
            and targets.
    '''
    config = config or DatasetViewConfig()

    # initial catalog view
    view = normalizer.read_dataset_manifest(
        inputs,
        head_pxs_thres=config.head_pxs_thres,
        focal_head=config.focal_head,
        non_overlapping_test_grid=config.non_overlapping_test_grid,
    )

    schema = inputs.schema

    # resolve feature channels
    features = semantics.resolve_feature_channels(schema, config.features)

    # resolve target heads
    targets = semantics.resolve_target_heads(schema, config.targets)

    # resolve focal head from target heads and config
    focal_head = semantics.resolve_focal_head(targets, config.focal_head)

    # enriched manifest with class counts
    enriched = _enrich_datablocks_manifest(view, schema, targets, focal_head)

    return DatasetView(
        focal_head=focal_head,
        crs=canvas_crs,
        transform=canvas_transform,
        manifest=enriched,
        features=features,
        targets=targets,
    )


# ----- private helpers
def _enrich_datablocks_manifest(
    datablocks_manifest: normalizer.NormalizedDataBlocksView,
    data_schema: geo_core.DatasetSchema,
    targets: semantics.TargetHeadsContext,
    focal_head: str,
) -> normalizer.NormalizedDataBlocksView:
    '''Enrich catalog view with derived class counts for focal head.'''
    row_size = data_schema['tensor_shapes']['image']['H']
    col_size = data_schema['tensor_shapes']['image']['W']

    updated_raw_counts: dict[tuple[int, int], dict[str, list[int]]] = {}
    valid_counts: dict[tuple[int, int], list[int]] = {}
    base_counts: dict[tuple[int, int], list[int]] = {}

    for coord, raw_counts in datablocks_manifest.raw_class_counts.items():
        head_counts = semantics.derive_head_class_counts(targets, raw_counts)
        updated_raw_counts[coord] = head_counts
        if focal_head in head_counts:
            counts = head_counts[focal_head]
            valid_counts[coord] = counts
            if coord[0] % col_size == 0 and coord[1] % row_size == 0:
                base_counts[coord] = counts

    return dataclasses.replace(
        datablocks_manifest,
        raw_class_counts=updated_raw_counts,
        valid_class_counts=valid_counts,
        base_class_counts=base_counts,
    )
