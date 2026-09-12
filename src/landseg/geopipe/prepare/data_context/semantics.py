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


# ----- `FeatureSelection`
@dataclasses.dataclass(frozen=True)
class FeatureSelection:
    '''Selected feature band names and 0-based channel indices.'''
    names: tuple[str, ...]
    indices: tuple[int, ...]


# ----- `TargetHeadsContext`
@dataclasses.dataclass(frozen=True)
class TargetHeadsContext:
    '''
    Multi-head target hierarchy and reclassification specifications.
    '''
    target_reclass: dict[str, geo_core.LabelScheme | None]
    head_names: tuple[str, ...]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]
    num_classes: dict[str, int]
    class_names: dict[str, list[str]]
    ignore_classes: dict[str, list[int]]


# ----- `resolve_feature_channels`
def resolve_feature_channels(
    band_map: typing.Mapping[str, int],
    user_features_cfg: typing.Mapping[str, str] | list[str] | None,
    feature_schemes: typing.Mapping[
        str, typing.Mapping[str, list[str]]
    ] | None,
) -> FeatureSelection:
    '''
    Resolve active feature band names and 0-based channel indices.

    If `user_features_cfg` is empty or None, all available bands in
    `band_map` are selected in sequential order.

    Args:
        band_map: Mapping of band names to 0-based channel indices.
        user_features_cfg: User selection by scheme name, band list,
            or None to enable all bands.
        feature_schemes: Dataset manifest feature schemes metadata.

    Returns:
        A `FeatureSelection` containing band names and channel indices.
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


# ----- `resolve_target_heads`
def resolve_target_heads(
    label_names_map: typing.Mapping[str, list[str]] | typing.Sequence[str],
    user_targets_cfg: (
        typing.Mapping[str, str | geo_core.LabelScheme] | None
    ) = None,
    label_schemes: typing.Mapping[
        str, typing.Mapping[str, geo_core.LabelScheme]
    ] | None = None,
    label_ignore_cls: typing.Mapping[str, list[int]] | None = None,
) -> TargetHeadsContext:
    '''
    Resolve multi-head target hierarchy and reclassifications.

    Builds head relationships including base heads, child slices, and
    grouping layers according to user configuration and label schemes.

    Args:
        label_names_map: Mapping of label layer name to class names
            or sequence of label layer names.
        user_targets_cfg: User target configuration per label layer.
        label_schemes: Dataset manifest label schemes metadata.
        label_ignore_cls: Optional mapping of layer name to ignore class
            IDs.

    Returns:
        A `TargetHeadsContext` detailing full multi-head topology.
    '''
    # normalize label names map and ignore classes
    if isinstance(label_names_map, typing.Mapping):
        names_map = {k: list(v) for k, v in label_names_map.items()}
    else:
        names_map = {k: [] for k in label_names_map}

    ignore_map = dict(label_ignore_cls) if label_ignore_cls else {}
    schemes_dict = label_schemes or {}
    user_cfg = user_targets_cfg or {}

    resolved_reclass: dict[str, geo_core.LabelScheme | None] = {}
    head_names: list[str] = []
    head_parent: dict[str, str | None] = {}
    head_parent_cls: dict[str, int | None] = {}
    num_classes: dict[str, int] = {}
    class_names: dict[str, list[str]] = {}
    ignore_classes: dict[str, list[int]] = {}

    for label_name in names_map:
        cfg = user_cfg.get(label_name)

        # resolve reclassification scheme for this layer
        reclass_scheme: geo_core.LabelScheme | None = None
        if cfg is None or cfg in ('raw', 'base', 'none'):
            reclass_scheme = None
        elif isinstance(cfg, str):
            _require_key(label_name, schemes_dict, 'source schemes')
            current_schemes = schemes_dict[label_name]
            _require_key(cfg, current_schemes, 'label schemes')
            reclass_scheme = current_schemes[cfg]
        elif isinstance(cfg, dict):
            if (
                'reclass' not in cfg
                or not isinstance(cfg['reclass'], dict)
            ):
                raise ValueError(
                    f'Target reclassification dict for "{label_name}" '
                    'must contain a "reclass" mapping'
                )
            reclass_scheme = cfg
        else:
            raise TypeError(
                f'Invalid target config type for "{label_name}": '
                f'expected str or dict, got {type(cfg)}'
            )

        resolved_reclass[label_name] = reclass_scheme
        base_classes = names_map[label_name]
        base_ignore = ignore_map.get(label_name, [255])
        primary_ignore = base_ignore[0] if base_ignore else 255

        # 1. base head
        head_names.append(label_name)
        head_parent[label_name] = None
        head_parent_cls[label_name] = None
        num_classes[label_name] = len(base_classes)
        class_names[label_name] = list(base_classes)
        ignore_classes[label_name] = list(base_ignore)

        if not reclass_scheme or not reclass_scheme.get('reclass'):
            continue

        reclass = reclass_scheme['reclass']
        reclass_name = reclass_scheme.get('reclass_name') or {}
        group_head_name = f'{label_name}_group'

        # 2. child slice heads
        for group_id, child_classes in reclass.items():
            grp_name = reclass_name.get(str(group_id))
            if grp_name:
                child_head_name = f'{label_name}_{grp_name}'
            else:
                child_head_name = f'{label_name}_sub{group_id}'

            head_names.append(child_head_name)
            head_parent[child_head_name] = group_head_name
            head_parent_cls[child_head_name] = int(group_id)
            num_classes[child_head_name] = len(child_classes)

            # resolve class names for child slice
            child_cnames: list[str] = []
            for cls_id in child_classes:
                if 1 <= cls_id <= len(base_classes):
                    child_cnames.append(base_classes[cls_id - 1])
                elif 0 <= cls_id < len(base_classes):
                    child_cnames.append(base_classes[cls_id])
                else:
                    child_cnames.append(f'class_{cls_id}')
            class_names[child_head_name] = child_cnames
            ignore_classes[child_head_name] = [primary_ignore]

        # 3. grouping head
        head_names.append(group_head_name)
        head_parent[group_head_name] = None
        head_parent_cls[group_head_name] = None
        num_classes[group_head_name] = len(reclass)

        group_cnames = [
            reclass_name.get(str(g), f'group_{g}') for g in reclass
        ]
        class_names[group_head_name] = group_cnames
        ignore_classes[group_head_name] = [primary_ignore]

    return TargetHeadsContext(
        target_reclass=resolved_reclass,
        head_names=tuple(head_names),
        head_parent=head_parent,
        head_parent_cls=head_parent_cls,
        num_classes=num_classes,
        class_names=class_names,
        ignore_classes=ignore_classes,
    )


# ----- `_require_key`
def _require_key(
    key: str,
    mapping: typing.Mapping,
    kind: str,
) -> None:
    '''Validate existence of a key within a mapping container.'''
    if key not in mapping:
        raise KeyError(
            f'Unknown {kind} "{key}". Available {kind}s: {list(mapping)}'
        )
