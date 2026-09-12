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

    return DatasetContext(
        catalog=blocks_view,
        features=feature_selection,
        targets=target_heads,
    )
