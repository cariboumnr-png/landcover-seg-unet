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
Dataset context orchestration utilities.

Provides a unified builder coordinating catalog reading, feature
channel selection, and target head hierarchy resolution into a single
self-contained dataset preparation context.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# third-party imports
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.data_context.catalog as catalog
import landseg.geopipe.prepare.data_context.semantics as semantics


# typing aliases
SchemaCtrl = artifacts.Controller[geo_core.DataSchema]


# ----- `DatasetContext`
@dataclasses.dataclass(frozen=True)
class DatasetContext:
    '''
    Unified dataset preparation context for partitioning and
    normalization.
    '''
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


# ----- `build_dataset_context`
def build_dataset_context(
    catalog_fpath: str,
    schema_fpath: str,
    catalog_config: catalog._CatalogViewConfig,
    user_features_cfg: typing.Mapping[str, str] | list[str] | None = None,
    user_targets_cfg: (
        typing.Mapping[str, str | geo_core.LabelScheme] | None
    ) = None,
) -> DatasetContext:
    '''
    Build complete dataset context from catalog, schema, and config.

    Orchestrates catalog parsing, feature channel resolution, and
    multi-head target hierarchy construction into a unified container.
    Derives per-class pixel counts for all target heads in memory
    without reading block files or raster data from disk.

    Args:
        catalog_fpath: Path to canonical blocks catalog JSON.
        schema_fpath: Path to dataset schema JSON.
        catalog_config: Valid-pixel filtering and test configuration.
        user_features_cfg: Optional user feature band configuration.
        user_targets_cfg: Optional user target reclass configuration.

    Returns:
        A `DatasetContext` combining catalog, features, and targets.
    '''
    # load schema
    schema_ctrl = SchemaCtrl.load_json_or_fail(schema_fpath)
    schema = schema_ctrl.fetch()

    # read catalog blocks view using loaded schema
    blocks_view = catalog.read_catalog(
        catalog_fpath=catalog_fpath,
        data_schema=schema,
        config=catalog_config,
    )

    # resolve feature channels
    image_band_map = schema.get('io_conventions', {}).get(
        'image_band_map', {}
    )
    dataset_schemes = schema.get('dataset', {}).get('schemes', {})
    feature_selection = semantics.resolve_feature_channels(
        band_map=image_band_map,
        user_features_cfg=user_features_cfg,
        feature_schemes=dataset_schemes,
    )

    # resolve target heads
    labels_info = schema.get('labels', {})
    label_names = labels_info.get(
        'label_class_names', labels_info.get('label_names', {})
    )
    label_ignore = labels_info.get('label_ignore_cls', {})
    target_heads = semantics.resolve_target_heads(
        label_names_map=label_names,
        user_targets_cfg=user_targets_cfg,
        label_schemes=dataset_schemes,
        label_ignore_cls=label_ignore,
    )

    # resolve focal head from target heads and config
    focal_target = getattr(catalog_config, 'focal_target', None)
    focal_head = semantics.resolve_focal_head(target_heads, focal_target)

    # image shape to determine base grid coordinates
    image_shape = schema['tensor_shapes']['image']
    row_size, col_size = image_shape['H'], image_shape['W']

    # derive class counts for focal head in memory
    valid_counts: dict[tuple[int, int], list[int]] = {}
    base_counts: dict[tuple[int, int], list[int]] = {}

    for coord, raw_counts in blocks_view.raw_class_counts.items():
        head_counts = semantics.derive_head_class_counts(
            raw_counts, target_heads
        )
        if focal_head in head_counts:
            counts = head_counts[focal_head]
            valid_counts[coord] = counts
            if coord[0] % row_size == 0 and coord[1] % col_size == 0:
                base_counts[coord] = counts

    enriched_view = dataclasses.replace(
        blocks_view,
        focal_head=focal_head,
        valid_class_counts=valid_counts,
        base_class_counts=base_counts,
    )

    return DatasetContext(
        catalog=enriched_view,
        features=feature_selection,
        targets=target_heads,
    )
