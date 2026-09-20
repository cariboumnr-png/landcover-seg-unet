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

Public APIs:
    - FeatureSelection: selected band names and 0-based channel indices.
    - TargetHeadsContext: multi-head hierarchy and reclass specs.
    - resolve_feature_channels: resolve active feature bands.
    - resolve_target_heads: resolve multi-head target hierarchy.
    - resolve_focal_head: resolve focal target head for partitioning.
    - derive_head_class_counts: derive counts across all target heads.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.geopipe.core as geo_core


# ----- typing aliases
field = dataclasses.field


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class FeatureSelection:
    '''Selected feature band names and 0-based channel indices.'''
    names: tuple[str, ...]
    indices: tuple[int, ...]

    def __iter__(self) -> typing.Iterator[list[str] | list[int]]:
        return iter([list(self.names), list(self.indices)])


@dataclasses.dataclass(frozen=True)
class TargetHeadsContext:
    '''Multi-head target hierarchy and reclassification specs.'''
    head_names: list[str]
    head_parent: dict[str, str | None]
    head_parent_cls: dict[str, int | None]
    num_classes: dict[str, int]
    class_names: dict[str, list[str]]
    ignore_classes: dict[str, list[int]]
    resolved_reclass: dict[str, _ResolvedTargetReclass | None]


# ----- private dataclasses
@dataclasses.dataclass(frozen=True)
class _ResolvedTargetReclass:
    '''Canonical zero-based reclassification mapping.'''
    groups: dict[int, tuple[int, ...]]
    names: dict[int, str]

    @classmethod
    def from_scheme(
        cls, scheme: geo_core.LabelScheme
    ) -> _ResolvedTargetReclass:
        '''Build canonical 0-based reclassification from user scheme.'''
        reclass = scheme['reclass']
        names = scheme.get('reclass_name', {})
        return cls(
            groups={
                int(k) - 1: tuple(v - 1 for v in vv)
                for k, vv in reclass.items()
            },
            names={int(k) - 1: v for k, v in names.items()},
        )

    def __iter__(self) -> typing.Iterator[tuple[int, tuple[int, ...], str]]:
        for group_id, source_classes in self.groups.items():
            yield group_id, source_classes, self.names.get(group_id, '')

    def __getitem__(self, index: int) -> tuple[tuple[int, ...], str]:
        return self.groups[index], self.names[index]


@dataclasses.dataclass
class _TargetTopology:
    '''Internal mutable accumulator for multi-head target topology.'''
    head_names: list[str] = field(default_factory=list)
    head_parent: dict[str, str | None] = field(default_factory=dict)
    head_parent_cls: dict[str, int | None] = field(default_factory=dict)
    num_classes: dict[str, int] = field(default_factory=dict)
    class_names: dict[str, list[str]] = field(default_factory=dict)
    ignore_classes: dict[str, list[int]] = field(default_factory=dict)

    def to_context(
        self,
        resolved_reclass: dict[str, _ResolvedTargetReclass | None],
    ) -> TargetHeadsContext:
        '''Convert accumulated topology into TargetHeadsContext.'''
        return TargetHeadsContext(
            head_names=self.head_names,
            head_parent=self.head_parent,
            head_parent_cls=self.head_parent_cls,
            num_classes=self.num_classes,
            class_names=self.class_names,
            ignore_classes=self.ignore_classes,
            resolved_reclass=resolved_reclass,
        )


# ----- public functions
def resolve_feature_channels(
    data_schema: geo_core.DatasetSchema,
    user_features_cfg: typing.Mapping[str, str] | list[str] | None,
) -> FeatureSelection:
    '''
    Resolve active feature band names and 0-based channel indices.

    If `user_features_cfg` is empty or None, all available bands in
    `band_map` are selected in sequential order.

    Args:
        data_schema:
            catalog data schema containing band mapping and schemes.
        user_features_cfg:
            user selection by scheme name, band list, or None to enable
            all available bands.

    Returns:
        FeatureSelection:
            selected feature band names and 0-based channel indices.
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


def resolve_target_heads(
    data_schema: geo_core.DatasetSchema,
    user_targets_cfg: typing.Mapping[str, str | geo_core.LabelScheme] | None,
) -> TargetHeadsContext:
    '''
    Resolve multi-head target hierarchy and reclassifications.

    Builds head relationships including base heads, child slices, and
    grouping layers according to user configuration and label schemes.

    Args:
        data_schema:
            ingested dataset schema container.
        user_targets_cfg:
            user target configuration per label layer.

    Returns:
        TargetHeadsContext:
            resolved multi-head hierarchy context.
    '''
    user_cfg = dict(user_targets_cfg or {})
    label_name_map = data_schema['io_conventions']['label_band_map']
    label_schemes = data_schema['dataset']['label_schemes']
    labels_info = data_schema['labels']

    resolved_reclass: dict[str, _ResolvedTargetReclass | None] = {}
    topo = _TargetTopology()

    for name in label_name_map.keys():
        # 1. record base head
        topo.head_names.append(name)
        topo.head_parent[name] = None
        topo.head_parent_cls[name] = None
        topo.num_classes[name] = labels_info['label_num_cls'][name]
        topo.class_names[name] = list(labels_info['label_class_names'][name])
        base_ignore = labels_info['label_ignore_cls'][name]
        topo.ignore_classes[name] = list(base_ignore)

        # 2. resolve scheme
        layer_schemes = label_schemes.get(name, {})
        scheme = _resolve_layer_scheme(name, user_cfg.get(name), layer_schemes)
        if not scheme:
            resolved_reclass[name] = None
            continue

        # 3. canonical reclass & multi-head topology
        resolved_reclass[name] = _ResolvedTargetReclass.from_scheme(scheme)
        primary_ignore = base_ignore[0] if base_ignore else 255
        _add_reclass_heads(
            name,
            scheme,
            topo.class_names[name],
            primary_ignore,
            topo,
        )

    return topo.to_context(resolved_reclass)


def resolve_focal_head(
    target_heads: TargetHeadsContext,
    focal_target: str | None = None,
) -> str:
    '''
    Resolve active focal target head for partitioning.

    Matches direct head names, child slice or grouping class names,
    or falls back to the default grouping or primary head.

    Args:
        target_heads:
            resolved multi-head hierarchy context.
        focal_target:
            optional requested focal head or class name.

    Returns:
        str:
            the resolved focal head name string.
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


def derive_head_class_counts(
    target_heads: TargetHeadsContext,
    raw_class_counts: typing.Mapping[str, typing.Sequence[int]],
) -> dict[str, list[int]]:
    '''
    Derive per-class pixel counts for all target heads in memory.

    Given raw base label class counts for a block, computes exact class
    counts for all heads including child slices and grouping layers.

    Args:
        target_heads:
            resolved multi-head hierarchy and reclass schemes.
        raw_class_counts:
            mapping of base label name to raw per-block class counts.

    Returns:
        dict[str, list[int]]:
            mapping of head name to derived per-class pixel counts list.
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


# ----- private helpers
def _resolve_layer_scheme(
    name: str,
    cfg: str | geo_core.LabelScheme | None,
    available_schemes: typing.Mapping[str, geo_core.LabelScheme],
) -> geo_core.LabelScheme | None:
    '''Resolve and validate raw reclass scheme for one layer.'''
    if cfg is None or cfg in ('raw', 'base', 'none'):
        return None
    if isinstance(cfg, str):
        _require_key(cfg, available_schemes, 'label schemes')
        return available_schemes[cfg]
    if isinstance(cfg, dict):
        if (
            'reclass' not in cfg
            or not isinstance(cfg['reclass'], dict)
        ):
            raise ValueError(
                f'Target reclassification dict for "{name}" '
                'must contain a "reclass" mapping'
            )
        return cfg
    raise TypeError(
        f'Invalid target config type for "{name}": '
        f'expected str or dict, got {type(cfg)}'
    )


def _add_reclass_heads(
    name: str,
    scheme: geo_core.LabelScheme,
    base_classes: list[str],
    primary_ignore: int,
    topo: _TargetTopology,
) -> None:
    '''Append group and child slice heads to topology accumulator.'''
    reclass = scheme['reclass']
    reclass_name = scheme.get('reclass_name', {})

    # 1. group head (parent)
    group_head = f'{name}_group'
    topo.head_names.append(group_head)
    topo.head_parent[group_head] = None
    topo.head_parent_cls[group_head] = None
    topo.num_classes[group_head] = len(reclass)
    topo.ignore_classes[group_head] = [primary_ignore]
    topo.class_names[group_head] = [
        reclass_name.get(str(g), f'group_{g}') for g in reclass
    ]

    # 2. child slice heads (sub-groups)
    for group_id, child_classes in reclass.items():
        grp_name = reclass_name.get(str(group_id))
        child_head = (
            f'{name}_{grp_name}' if grp_name else f'{name}_sub{group_id}'
        )

        topo.head_names.append(child_head)
        topo.head_parent[child_head] = group_head
        topo.head_parent_cls[child_head] = int(group_id)
        topo.num_classes[child_head] = len(child_classes)
        topo.ignore_classes[child_head] = [primary_ignore]
        topo.class_names[child_head] = [
            base_classes[c - 1]
            if 1 <= c <= len(base_classes)
            else f'class_{c}'
            for c in child_classes
        ]


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
