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

Converts user-provided manifest dictionaries into canonical
`ManifestEntry` objects and validates category-specific metadata,
band mappings, categorical specifications, schemes, and taxonomy
references.
'''

# standard imports
from __future__ import annotations
import typing
import pathlib
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.taxonomy as taxonomy

# ----- public classes
class ManifestEntryNormalizer:
    '''
    Normalize and validate a single dataset manifest entry.

    This class transforms a user-provided manifest mapping into a
    canonical `ManifestEntry` object, validating required fields and
    resolving category-specific metadata such as schemes and taxonomy
    specifications.
    '''

    def __init__(self, entry: typing.Any):
        '''Initialize with the input object (expect a dictionary).'''
        self.entry = _require_dict(entry)

        self.name = self.entry.get('name')
        self.path = self.entry.get('path')
        self.band_mapping = self.entry.get('band_mapping')
        self.category = self.entry.get('category')
        self.cat_specs = self.entry.get('categorical_specs')
        self.schemes = self.entry.get('schemes')

    @property
    def normalized_entry(self) -> manifest.ManifestEntry:
        '''
        Return the normalized manifest entry.

        The returned mapping contains validated and canonicalized values
        suitable for downstream harmonization processing.
        '''
        name = _require_string(self.name)

        path = _require_path(self.path)

        band_mapping = self._normalize_band_mapping()

        category = self._normalize_category()

        if category in ['domain', 'domains']:
            categorical_specs = self._normalize_categorical_specs()
            if self.schemes:
                raise ValueError('Domain rasters should not define "schemes"')
            schemes = None
        elif category in ['label', 'labels']:
            categorical_specs = self._normalize_categorical_specs()
            if self.schemes:
                schemes = self._normalize_label_schemes(categorical_specs)
            else:
                schemes = None
        else:
            categorical_specs = None
            if self.schemes:
                schemes = self._normalize_feature_schemes(band_mapping)
            else:
                schemes = None

        normalized: manifest.ManifestEntry = {
            'name': name,
            'path': path,
            'band_mapping': band_mapping,
            'category': category,
            'categorical_specs': categorical_specs,
            'schemes': schemes
        }
        return normalized

    def validate(self) -> None:
        '''Validate the manifest entry.'''
        _ = self.normalized_entry

    def _normalize_band_mapping(self) -> dict[int, str]:
        mapping = _require_dict_w_str_values(self.band_mapping)
        normalized = {_require_int_w_min(k, 1): v for k, v in mapping.items()}
        if set(normalized.keys()) != set(range(1, len(normalized) + 1)):
            raise ValueError(
                f'Band mapping keys should be contiguous integers from 1, '
                f'got: {sorted(normalized.keys())}'
            )

        return normalized

    def _normalize_category(self) -> manifest.AllowedCategory:
        cat = _require_string(self.category)
        allowed = typing.get_args(manifest.AllowedCategory)
        if cat not in allowed:
            raise ValueError(f'Invalid category: {cat}, must be in {allowed}')

        return typing.cast(manifest.AllowedCategory, cat)

    def _normalize_categorical_specs(self) -> geo_core.CategoricalSpecs:
        specs = _require_dict(self.cat_specs)

        # init w mandatory fields
        index_base = _require_int_w_min(specs.get('index_base'), 0)
        num_cls = _require_int_w_min(specs.get('num_cls'), 1)
        ignore_cls = _require_int_list(specs.get('ignore_cls'))
        _specs: geo_core.CategoricalSpecs = {
            'index_base': index_base,
            'num_cls': num_cls,
            'ignore_cls': ignore_cls
        }

        # add optional fields
        class_name = specs.get('class_name')
        if class_name is not None:
            _specs['class_name'] = _require_dict_w_str_values(class_name)

        color_map = specs.get('color_map')
        if color_map is not None:
            _specs['color_map'] = _require_dict(color_map)
            for rgb in _specs['color_map'].values():
                _require_int_list(rgb, 3) # rgb values

        taxa = specs.get('taxonomy')
        if taxa is not None:
            if not class_name:
                raise ValueError('Class names not provided for lookup')

            taxa = _require_dict(taxa)
            profile = _require_string(taxa.get('profile'))
            taxa_specs = taxonomy.validate_specs(profile, class_name, num_cls)
            _specs['taxonomy'] = taxa_specs

        return _specs

    def _normalize_feature_schemes(
        self,
        band_mapping: dict[int, str]
    ) -> manifest.FeatureSchemes:
        schemes = _require_dict(self.schemes)

        valid_bands = set(band_mapping.values())
        for name, bands in schemes.items():
            if not isinstance(bands, list) or not bands:
                raise ValueError(
                    f'Feature scheme "{name}" must be non-empty list of '
                    f'band names, got: {bands}'
                )
            for b in bands:
                if not isinstance(b, str) or b not in valid_bands:
                    raise ValueError(
                        f'Band "{b}" in feature scheme "{name}" is not in '
                        f'band_mapping: {sorted(valid_bands)}'
                    )
        return schemes

    def _normalize_label_schemes(
        self,
        cat_specs: manifest.CategoricalSpecs
    ) -> manifest.LabelSchemes:
        schemes = _require_dict(self.schemes)

        index_base = cat_specs['index_base']
        num_cls = cat_specs['num_cls']
        valid_range = set(range(index_base, index_base + num_cls))

        for name, data in schemes.items():
            data = _require_dict(data)

            reclass = _require_dict(data.get('reclass'))
            for grp_id, cls_ids in reclass.items():
                cls_ids = _require_int_list(cls_ids)
                for cid in cls_ids:
                    if cid not in valid_range:
                        raise ValueError(
                            f'Class ID {cid} in label scheme "{name}" group '
                            f'"{grp_id}" is outside valid class range '
                            f'[{index_base}..{index_base + num_cls - 1}]'
                        )

            reclass_name = _require_dict_w_str_values(data.get('reclass_name'))
            for grp_id in reclass_name:
                if grp_id not in reclass:
                    raise ValueError(
                        f'Group "{grp_id}" in reclass_name does not exist in '
                        f'reclass of label scheme "{name}"'
                    )

        return schemes


# ----- private helpers
def _require_dict(d: typing.Any) -> dict:
    if not isinstance(d, dict):
        raise TypeError(f'Input {d} must be a dict, got type {type(d)}')
    if len(d) == 0:
        raise ValueError('Input dict is empty')
    return typing.cast(dict, d)


def _require_string(s: typing.Any) -> str:
    if not isinstance(s, str):
        raise TypeError(f'Input {s} must be a string, got type {type(s)}')
    if len(s) == 0:
        raise ValueError('Input string is empty')
    return typing.cast(str, s)


def _require_path(p: typing.Any) -> str:
    try:
        pp = pathlib.Path(p)
        return str(pp)
    except (TypeError, ValueError) as e:
        raise ValueError('Input is not a valid path') from e


def _require_int_w_min(n: typing.Any, min_value: int) -> int:
    try:
        n = int(n)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Input {n!r} must be convertible to an integer"
        ) from e
    if n < min_value:
        raise ValueError(f'Input integer {n} < min value {min_value}')
    return typing.cast(int, n)


def _require_int_list(l: typing.Any, nint: int | None = None) -> list[int]:
    if not isinstance(l, list):
        raise TypeError(f'Input {l} must be a list, got type {type(l)}')
    if not all(isinstance(x, int) for x in l):
        raise ValueError(f'Input {l} must be a list of ints, got: {l}')
    if nint and len(l) != nint:
        raise ValueError(f'Input {l} must have {nint} integers, got: {len(l)}')
    return typing.cast(list[int], l)


def _require_dict_w_str_values(d: typing.Any) -> dict[str, str]:
    _require_dict(d)
    if not all(isinstance(s, str) for s in d.values()):
        raise TypeError(f'Not all values in input dict are a string from {d}')
    return typing.cast(dict[str, str], d)
