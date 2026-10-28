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
Manifest entry normalization and validation utilities.

This module converts user-provided manifest dictionaries into canonical
`ManifestEntry` objects and validates category-specific metadata,
band mappings, categorical specifications, schemes, and taxonomy
references.

Public APIs:
    - `AllowedCategory`: Type alias for valid raster categories.
    - `ManifestEntry`: TypedDict defining per-raster configuration.
    - `normalize_manifest_entry`: Validate and normalize manifest entry.
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.utils.validator as val


# ----- typing aliases
AllowedCategory: typing.TypeAlias = typing.Literal[
    'domains',
    'domain',
    'features',
    'feature',
    'labels',
    'label',
]


# ----- public types
class ManifestEntry(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: pathlib.Path
    band_mapping: dict[int, str]
    category: AllowedCategory
    categorical_specs: geo_core.CategoricalSpec | None
    schemes: dict[str, geo_core.LabelScheme] | dict[str, tuple[str, ...]] | None


# ----- public functions
def normalize_manifest_entry(entry: typing.Any) -> ManifestEntry:
    '''
    Normalize and validate a single dataset manifest entry.

    Transforms a user-provided manifest mapping into a canonical
    `ManifestEntry` object, validating required fields and resolving
    category-specific metadata such as schemes and taxonomy
    specifications.

    Args:
        entry:
            Raw manifest entry mapping from user configuration.

    Returns:
        ManifestEntry:
            Canonical validated manifest entry.
    '''
    _entry = val.require_mapping(entry)
    name = val.require_string(_entry.get('name'))
    path = val.require_file(_entry.get('path'), missing_ok=False)
    band_mapping = _normalize_band_mapping(_entry.get('band_mapping'))
    category = _normalize_category(_entry.get('category'))
    raw_schemes = _entry.get('schemes')

    match category:
        case 'domain' | 'domains':
            if raw_schemes:
                raise ValueError('Domain rasters should not define "schemes"')
            cat_specs = _normalize_categorical(_entry.get('categorical_specs'))
            schemes = None

        case 'label' | 'labels':
            cat_specs = _normalize_categorical(_entry.get('categorical_specs'))
            schemes = (
                _normalize_label_schemes(raw_schemes, cat_specs)
                if raw_schemes
                else None
            )

        case 'feature' | 'features':
            cat_specs = None
            schemes = (
                _normalize_feature_schemes(raw_schemes, band_mapping)
                if raw_schemes
                else None
            )

    return {
        'name': name,
        'path': path,
        'band_mapping': band_mapping,
        'category': category,
        'categorical_specs': cat_specs,
        'schemes': schemes,
    }


# ----- private helpers
def _normalize_category(category: typing.Any) -> AllowedCategory:
    '''Validate and cast raster category string.'''
    allowed = typing.get_args(AllowedCategory)
    return typing.cast(
        AllowedCategory,
        val.require_value_from(category, allowed, str),
    )


def _normalize_band_mapping(raw: typing.Any) -> dict[int, str]:
    '''Validate and convert band mapping keys to contiguous 1-based ints.'''
    mapping = val.require_mapping_str_v(raw)
    normalized: dict[int, str] = {}
    for k, v in mapping.items():
        k_int = val.require_numeric_min(int(k), 1, int)
        # int conversion will throw if not an int and propegate to compiler.py
        normalized[k_int] = v
    if set(normalized.keys()) != set(range(1, len(normalized) + 1)):
        raise ValueError(
            f'Band mapping keys should be contiguous integers from 1, '
            f'got: {sorted(normalized.keys())}'
        )
    return normalized


def _normalize_categorical(raw: typing.Any) -> geo_core.CategoricalSpec:
    '''Validate categorical raster specifications.'''
    raw_dict = val.require_mapping(raw)
    specs = _init_categorical_specs(raw_dict)

    cls_names = raw_dict.get('class_name')
    if cls_names is not None:
        specs['class_name'] = val.require_mapping_str_v(cls_names)

    color_map = raw_dict.get('color_map')
    if color_map is not None:
        cmap = val.require_mapping(color_map)
        validated_cmp: dict[str, tuple[int, int, int]] = {}
        for k, rgb in cmap.items():
            rgb_t = val.require_numeric_sequence(rgb, int, 3)
            r = val.require_numeric_range(rgb_t[0], (0, 255), int)
            g = val.require_numeric_range(rgb_t[1], (0, 255), int)
            b = val.require_numeric_range(rgb_t[2], (0, 255), int)
            validated_cmp[str(k)] = (r, g, b)
        specs['color_map'] = validated_cmp

    taxanomy = raw_dict.get('taxonomy')
    if taxanomy is not None:
        if not cls_names:
            raise ValueError('Class names not provided for lookup')
        taxa_dict = val.require_mapping(taxanomy)
        profile = val.require_string(taxa_dict.get('profile'))
        specs['taxonomy'] = {'profile': profile}

    return specs


def _init_categorical_specs(raw: typing.Any) -> geo_core.CategoricalSpec:
    '''Initialize categorical specs with required fields.'''
    specs = val.require_mapping(raw)
    base = specs.get('index_base')
    num_cls = specs.get('num_cls')
    ignore = specs.get('ignore_cls')
    return {
        'index_base': val.require_numeric_min(base, 0, int),
        'num_cls': val.require_numeric_min(num_cls, 1, int),
        'ignore_cls': val.require_numeric_sequence(ignore, int),
    }


def _normalize_feature_schemes(
    raw_schemes: typing.Any,
    band_mapping: dict[int, str],
) -> dict[str, tuple[str, ...]]:
    '''Validate named feature band groupings.'''
    schemes = val.require_mapping(raw_schemes)
    valid_bands = tuple(band_mapping.values())
    normalized: dict[str, tuple[str, ...]] = {}

    for name, bands in schemes.items():
        b_seq = val.require_string_sequence(bands)
        if not b_seq:
            raise ValueError(
                f'Feature scheme "{name}" must be non-empty list of '
                f'band names, got: {bands}'
            )
        for b in b_seq:
            val.require_value_from(b, valid_bands, str)
        normalized[name] = b_seq
    return normalized


def _normalize_label_schemes(
    raw_schemes: typing.Any,
    cat_specs: geo_core.CategoricalSpec,
) -> dict[str, geo_core.LabelScheme]:
    '''Validate named label reclassification schemes.'''
    schemes = val.require_mapping(raw_schemes)
    index_base = cat_specs['index_base']
    num_cls = cat_specs['num_cls']
    cls_range = (index_base, index_base + num_cls - 1)

    normalized: dict[str, geo_core.LabelScheme] = {}
    for k, v in schemes.items():
        name = val.require_string(str(k))
        normalized[name] = _normalize_label_scheme(v, cls_range)
    return normalized


def _normalize_label_scheme(
    raw_scheme: typing.Any,
    cls_range: tuple[int, int],
) -> geo_core.LabelScheme:
    '''Validate a single label reclassification scheme.'''
    scheme_dict = val.require_mapping(raw_scheme)
    reclass = val.require_mapping(scheme_dict.get('reclass'))

    norm_reclass: dict[str, tuple[int, ...]] = {}
    for grp_id, ids in reclass.items():
        grp_key = val.require_string(str(grp_id))
        class_ids = val.require_numeric_sequence(ids)
        norm_reclass[grp_key] = tuple(
            val.require_numeric_range(i, cls_range, int) for i in class_ids
        )

    raw_names = scheme_dict.get('reclass_name')
    raw_names = val.require_mapping_str_v(raw_names)
    valid_groups = tuple(norm_reclass.keys())

    reclass_name: dict[str, str] = {}
    for grp_id, grp_label in raw_names.items():
        grp_key = val.require_value_from(str(grp_id), valid_groups, str)
        reclass_name[grp_key] = grp_label

    return {
        'reclass': norm_reclass,
        'reclass_name': reclass_name,
    }
