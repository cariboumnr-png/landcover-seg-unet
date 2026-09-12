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
Resolution utilities for feature channels and target reclassification.

Parses user-specified preparation configurations (named schemes or
inline overrides) against ingested catalog schemas to resolve active
input feature channels and multi-head target hierarchies.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.geopipe.core as geo_core


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class DatasetSemantics:
    '''doc'''
    features: FeatureSelection
    target: TargetSemantics


@dataclasses.dataclass(frozen=True)
class FeatureSelection:
    '''doc'''
    names: tuple[str, ...]
    indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TargetSemantics:
    '''doc'''
    source_label: str
    class_mapping: typing.Mapping[int, int]
    class_names: typing.Mapping[int, str]
    ignore_classes: frozenset[int]


# ----- public functions
def resolve_feature_channels(
    band_map: typing.Mapping[str, int],
    user_features_cfg: typing.Mapping[str, str] | list[str] | None,
    feature_schemes: typing.Mapping[str, typing.Mapping[str, list[str]]] | None,
) -> FeatureSelection:
    '''
    Resolve active feature band names and 0-based channel indices.

    If `user_features_cfg` is empty or None, all available bands in
    `band_map` are selected in sequential order.

    Example user feature config:
    mapping:
    ```
    {
        'sentinel2': 'rgb',
        'topo': 'all'
    }
    ```
    bands:
    ```
    ['red', 'green', 'blue', 'dem']
    ```

    Exmaple schemes:
    ```
    {
        'sentinel2': {
            'rgb': [...],
            ...
        },
        'topo': {
            'slope': [...],
        }
    }
    ```

    Args:
        band_map:
            Mapping of lower-case band names to 0-based channel indices
            in the ingested data blocks.
        user_features_cfg:
            Mapping of data source type a scheme name, an explicit list of
            band names, or `None` to enable all bands.
        feature_schemes:
            Mapping of data source type (e.g., sentinel2, topo) to its
            named schemes dictionary from dataset manifest metadata.

    Returns:
        A tuple of (selected_band_names, selected_channel_indices).
    '''
    # no feature selection -> use every ingested band.
    if not user_features_cfg:
        names = sorted(band_map, key=lambda k: band_map[k])
        return FeatureSelection(
            names=tuple(names),
            indices=tuple(band_map[name] for name in names),
        )

    # explicit band selection.
    if isinstance(user_features_cfg, list):
        selected = user_features_cfg

    # named scheme selection.
    else:
        if feature_schemes is None:
            raise ValueError(
                'Feature schemes are required when feature schemes '
                'are selected by name.'
            )

        selected = []
        for src_type, schema_name in user_features_cfg.items():
            _require_key(src_type, feature_schemes, 'feature source')
            schemes = feature_schemes[src_type]

            _require_key(schema_name, schemes, 'feature scheme')
            selected.extend(schemes[schema_name])

    # validate and deduplicate selected bands while preserving order.
    names = []
    seen = set()

    for band in selected:
        _require_key(band, band_map, 'feature band')

        if band not in seen:
            names.append(band)
            seen.add(band)

    return FeatureSelection(
        names=tuple(names),
        indices=tuple(band_map[name] for name in names),
    )


def resolve_target_heads(
    label_names_map: typing.Mapping[str, list[str]] | typing.Sequence[str],
    user_targets_cfg: typing.Mapping[str, str | geo_core.LabelScheme] | None,
    label_schemes: typing.Mapping[str, typing.Mapping[str, geo_core.LabelScheme]] | None,
) -> dict[str, geo_core.LabelScheme | None]:
    '''
    Resolve active reclassification settings per target label layer.

    Args:
        label_names_map:
            Mapping of label layer name to list of class names or
            sequence of label layer names.
        user_targets_cfg:
            Mapping of label name to scheme name or explicit
            reclassification specification dictionary.
        label_schemes:
            Mapping of label name to its named schemes.

    Returns:
        Mapping of label layer name to resolved LabelScheme or None.
    '''
    resolved: dict[str, geo_core.LabelScheme | None] = {}
    if not user_targets_cfg:
        return {k: None for k in label_names_map}

    schemes_dict = label_schemes or {}

    for label_name in label_names_map:
        schema_name = user_targets_cfg.get(label_name)

        if schema_name is None:
            resolved[label_name] = None
            continue

        if isinstance(schema_name, str):
            _require_key(label_name, schemes_dict, 'source schemes')
            current_label_schemes = schemes_dict[label_name]
            _require_key(schema_name, current_label_schemes, 'label schemes')
            resolved[label_name] = current_label_schemes[schema_name]

        elif isinstance(schema_name, dict):
            if (
                'reclass' not in schema_name or
                not isinstance(schema_name['reclass'], dict)
            ): # simple structural check
                raise ValueError(
                    f'Target reclassification dict for "{label_name}" must '
                    'contain a "reclass" mapping'
                )
            resolved[label_name] = schema_name

        else:
            raise TypeError(
                f'Invalid target config type for "{label_name}": '
                f'expected str or dict, got {type(schema_name)}'
            )

    return resolved


# ----- private helpers
def _require_key(
        key: str,
        mapping: typing.Mapping,
        kind: str,
    ) -> None:
    if key not in mapping:
        raise KeyError(
            f'Unknown {kind} "{key}". Available {kind}s: {list(mapping)}'
        )
