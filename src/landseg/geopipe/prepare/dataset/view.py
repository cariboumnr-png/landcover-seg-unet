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
    - `DatasetViewParameters`: configuration parameters for dataset view.
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
import landseg.geopipe.prepare.dataset.catalog as catalog
import landseg.geopipe.prepare.dataset.semantics as semantics


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class DatasetViewParameters:
    '''Configuration parameters for dataset view compilation.'''
    valid_pxs: dict[str, float] = dataclasses.field(default_factory=dict)
    focal_target: str | None = None
    test_catalog: str | None = None
    non_overlapping_test_grid: bool = True
    features: typing.Mapping[str, str] | list[str] | None = None
    targets: typing.Mapping[str, str | geo_core.LabelScheme] | None = None


@dataclasses.dataclass(frozen=True)
class DatasetView:
    '''Unified dataset view for partitioning and statistics.'''
    catalog: catalog.DataBlocksView
    features: semantics.FeatureSelection
    targets: semantics.TargetHeadsContext

    @property
    def focal_head(self) -> str:
        '''Active focal head name for partitioning.'''
        return self.catalog.focal_head

    @property
    def base_class_counts(self) -> dict[tuple[int, int], list[int]]:
        '''Base grid block per-class counts for active focal head.'''
        return self.catalog.base_class_counts

    @property
    def valid_class_counts(self) -> dict[tuple[int, int], list[int]]:
        '''Valid block per-class counts for active focal head.'''
        return self.catalog.valid_class_counts

    @property
    def valid_blocks(self) -> dict[tuple[int, int], str]:
        '''Mapping of block coordinate to file path.'''
        return self.catalog.valid_blocks

    @property
    def external_test_blocks(self) -> list[str] | None:
        '''Optional external holdout test block file paths.'''
        return self.catalog.external_test_blocks

    @property
    def crs(self) -> str:
        '''Coordinate reference system string of the dataset canvas.'''
        return self.catalog.crs

    @property
    def transform(self) -> rasterio.transform.Affine:
        '''Affine transform of the dataset canvas.'''
        return self.catalog.transform


# ----- public functions
def build_dataset_view(
    catalog_fpath: str,
    schema: geo_core.DatasetSchema,
    parameters: DatasetViewParameters | None = None,
    *,
    canvas_crs: str | None = None,
    canvas_transform: rasterio.transform.Affine | None = None,
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
    params = parameters or DatasetViewParameters()

    # initial catalog view
    view = catalog.read_catalog(
        catalog_fpath,
        schema,
        valid_pxs=params.valid_pxs,
        focal_target=params.focal_target,
        test_catalog=params.test_catalog,
        non_overlapping_test_grid=params.non_overlapping_test_grid,
        canvas_crs=canvas_crs,
        canvas_transform=canvas_transform,
    )

    # resolve feature channels
    feature_selection = semantics.resolve_feature_channels(
        schema, params.features
    )

    # resolve target heads
    targets = semantics.resolve_target_heads(schema, params.targets)

    # resolve focal head from target heads and config
    focal_head = semantics.resolve_focal_head(targets, params.focal_target)

    # enriched catalog view with class counts
    enriched_view = _enrich_view_w_class_counts(
        view, schema, targets, focal_head
    )

    return DatasetView(
        catalog=enriched_view,
        features=feature_selection,
        targets=targets,
    )


# ----- private helpers
def _enrich_view_w_class_counts(
    catalog_view: catalog.DataBlocksView,
    data_schema: geo_core.DatasetSchema,
    targets: semantics.TargetHeadsContext,
    focal_head: str,
) -> catalog.DataBlocksView:
    '''Enrich catalog view with derived class counts for focal head.'''
    row_size = data_schema['tensor_shapes']['image']['H']
    col_size = data_schema['tensor_shapes']['image']['W']

    updated_raw_counts: dict[tuple[int, int], dict[str, list[int]]] = {}
    valid_counts: dict[tuple[int, int], list[int]] = {}
    base_counts: dict[tuple[int, int], list[int]] = {}

    for coord, raw_counts in catalog_view.raw_class_counts.items():
        head_counts = semantics.derive_head_class_counts(targets, raw_counts)
        updated_raw_counts[coord] = head_counts
        if focal_head in head_counts:
            counts = head_counts[focal_head]
            valid_counts[coord] = counts
            if coord[0] % col_size == 0 and coord[1] % row_size == 0:
                base_counts[coord] = counts

    return dataclasses.replace(
        catalog_view,
        focal_head=focal_head,
        raw_class_counts=updated_raw_counts,
        valid_class_counts=valid_counts,
        base_class_counts=base_counts,
    )
