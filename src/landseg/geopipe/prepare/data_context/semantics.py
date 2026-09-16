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

    def __iter__(self) -> typing.Iterator[list[str] | list[int]]:
        return iter([list(self.names), list(self.indices)])


@dataclasses.dataclass(frozen=True)
class ResolvedTargetReclass:
    '''Canonical zero-based reclassification mapping.'''
    groups: dict[int, tuple[int, ...]]
    names: dict[int, str]

    def __iter__(self) -> typing.Iterator[tuple[int, tuple[int, ...], str]]:
        for group_id, source_classes in self.groups.items():
            yield group_id, source_classes, self.names.get(group_id, '')

    def __getitem__(self, index: int) -> tuple[tuple[int, ...], str]:
        return self.groups[index], self.names[index]


# ----- `TargetHeadsContext`
@dataclasses.dataclass(frozen=True)
class TargetHeadsContext:
    '''
    Multi-head target hierarchy and reclassification specifications.
    '''
    head_names: list[str]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]
    num_classes: dict[str, int]
    class_names: dict[str, list[str]]
    ignore_classes: dict[str, list[int]]
    resolved_reclass: dict[str, ResolvedTargetReclass | None]


# ----- `resolve_feature_channels`
def resolve_feature_channels(
    data_schema: geo_core.DataSchema,
    user_features_cfg: typing.Mapping[str, str] | list[str] | None,
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
    # fetch from data schema
    band_map = data_schema['io_conventions']['image_band_map']
    feature_schemes = data_schema['dataset']['image_schemes']
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
    data_schema: geo_core.DataSchema,
    user_targets_cfg: typing.Mapping[str, str | geo_core.LabelScheme] | None,
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
    # normalize user label reclass schemes
    user_targets_cfg = dict(user_targets_cfg or {})

    # fetach from data schema
    label_name_map = data_schema['io_conventions']['label_band_map']
    label_schemes = data_schema['dataset']['label_schemes']

    labels_info = data_schema['labels']
    # from labels_info
    ignore_classes = labels_info['label_ignore_cls']

    # init
    resolved_reclass: dict[str, ResolvedTargetReclass | None] = {}
    head_names: list[str] = []
    head_parent: dict[str, str | None] = {}
    head_parent_cls: dict[str, int | None] = {}
    num_classes: dict[str, int] = {}
    class_names: dict[str, list[str]] = {}

    for name in label_name_map.keys(): # 0-based

        # record base head info
        head_names.append(name)
        head_parent[name] = None
        head_parent_cls[name] = None
        num_classes[name] = labels_info['label_num_cls'][name]
        ignore_classes[name] = labels_info['label_ignore_cls'][name]
        class_names[name] = labels_info['label_class_names'][name]
        resolved_reclass[name] = None

        # resolve reclassification scheme for this layer
        reclass_scheme = user_targets_cfg.get(name)

        if reclass_scheme is None:
            pass

        elif isinstance(reclass_scheme, str):
            _require_key(reclass_scheme, label_schemes[name], 'label schemes')
            reclass_scheme = label_schemes[name][reclass_scheme]

        elif isinstance(reclass_scheme, dict):
            if (
                'reclass' not in reclass_scheme
                or not isinstance(reclass_scheme['reclass'], dict)
            ):
                raise ValueError(
                    f'Target reclassification dict for "{name}" '
                    'must contain a "reclass" mapping'
                )

        else:
            raise TypeError(
                f'Invalid target config type for "{name}": '
                f'expected str or dict, got {type(reclass_scheme)}'
            )

        if not reclass_scheme:
            continue # no reclassification for this head

        reclass = reclass_scheme['reclass']
        reclass_name = reclass_scheme.get('reclass_name', {})

        # resolve to 0-based
        resolved_reclass[name] = ResolvedTargetReclass(
            groups={
                int(k) - 1: tuple(v - 1 for v in vv)
                for k, vv in reclass.items()
            },
            names={
                int(k) - 1: v
                for k, v in reclass_name.items()
            }
        )

        # proceed to reclassification process
        base_classes = labels_info['label_class_names'][name]
        base_ignore = labels_info['label_ignore_cls'][name]
        primary_ignore = base_ignore[0] if base_ignore else 255

        group_head_name = f'{name}_group'

        # add grouped head (parent)
        head_names.append(group_head_name)
        head_parent[group_head_name] = None
        head_parent_cls[group_head_name] = None
        num_classes[group_head_name] = len(reclass)
        ignore_classes[group_head_name] = [primary_ignore]
        class_names[group_head_name] = [reclass_name.get(str(g), f'group_{g}') for g in reclass]

        # add sub-group heads (child)
        for group_id, child_classes in reclass.items():
            grp_name = reclass_name.get(str(group_id))
            if grp_name:
                child_head_name = f'{name}_{grp_name}'
            else:
                child_head_name = f'{name}_sub{group_id}'

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

    return TargetHeadsContext(
        head_names=head_names,
        head_parent=head_parent,
        head_parent_cls=head_parent_cls,
        num_classes=num_classes,
        class_names=class_names,
        ignore_classes=ignore_classes,
        resolved_reclass=resolved_reclass,
    )


# ----- `derive_head_class_counts`
def derive_head_class_counts(
    target_heads: TargetHeadsContext,
    raw_class_counts: typing.Mapping[str, typing.Sequence[int]],
) -> dict[str, list[int]]:
    '''
    Derive per-class pixel counts for all target heads in memory.

    Given raw base label class counts for a block, computes exact class
    counts for all heads including child slices and grouping layers.

    Args:
        raw_class_counts: Mapping of base label name to raw per-block
            class counts.
        target_heads: Resolved multi-head hierarchy and reclass schemes.

    Returns:
        Mapping of head name to derived per-class pixel counts list.
    '''
    derived: dict[str, list[int]] = {}

    for base_name in target_heads.head_names:
        if base_name in raw_class_counts:
            raw = list(raw_class_counts[base_name])
            derived[base_name] = raw
            reclass = target_heads.resolved_reclass.get(base_name)
            if reclass is None:
                continue

            group_head_name = f'{base_name}_group'

            # 1. child slices
            for group_id, child_classes, grp_name in reclass:
                child_head = (
                    f'{base_name}_{grp_name}'
                    if grp_name
                    else f'{base_name}_sub{group_id}'
                )
                child_counts = [0] * len(child_classes)
                for k, cls_id in enumerate(child_classes):
                    if 0 <= cls_id < len(raw):
                        child_counts[k] = raw[cls_id]
                derived[child_head] = child_counts

            # 2. grouping head
            max_gid = max(reclass.groups.keys())    # keys are 0-based
            group_counts = [0] * (max_gid + 1)      # so need to + 1
            for group_id, child_classes in reclass.groups.items():
                gid = int(group_id)
                group_counts[gid] = sum(
                    raw[c] for c in child_classes if 0 <= c < len(raw)
                )
            derived[group_head_name] = group_counts

    return derived


# ----- `resolve_focal_head`
def resolve_focal_head(
    target_heads: TargetHeadsContext,
    focal_target: str | None = None,
) -> str:
    '''
    Resolve active focal target head for partitioning.

    Matches direct head names, child slice or grouping class names,
    or falls back to the default grouping or primary head.

    Args:
        target_heads: Resolved multi-head hierarchy context.
        focal_target: Optional requested focal head or class name.

    Returns:
        The resolved focal head name string.
    '''
    if not target_heads.head_names:
        raise ValueError('No target heads available in context')

    if focal_target is None:
        # prefer grouping head if reclassified, else primary base head
        for h in target_heads.head_names:
            if h.endswith('_group'):
                return h
        return target_heads.head_names[0]

    # 1. exact match with head name
    if focal_target in target_heads.head_names:
        return focal_target

    # 2. match against class names across heads
    for head_name, cnames in target_heads.class_names.items():
        if focal_target in cnames:
            return head_name

    # 3. match against child slice group prefix/suffix
    for head_name in target_heads.head_names:
        if head_name.endswith(f'_{focal_target}'):
            return head_name

    raise KeyError(
        f'Focal target "{focal_target}" not found in available target '
        f'heads {list(target_heads.head_names)} or class names'
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
