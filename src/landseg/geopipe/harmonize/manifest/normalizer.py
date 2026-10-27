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
    - `ManifestEntryNormalizer`: Normalize and validate a manifest entry.
'''

# standard imports
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.manifest.schema as schema
import landseg.utils.validator as val


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
        self.entry = val.require_mapping(entry, 'manifest_entry')

        self.name = self.entry.get('name')
        self.path = self.entry.get('path')
        self.band_mapping = self.entry.get('band_mapping')
        self.category = self.entry.get('category')
        self.cat_specs = self.entry.get('categorical_specs')
        self.schemes = self.entry.get('schemes')

    @property
    def normalized_entry(self) -> schema.ManifestEntry:
        '''
        Return the normalized manifest entry.

        The returned mapping contains validated and canonicalized values
        suitable for downstream harmonization processing.
        '''
        name = val.require_string(self.name, 'entry_name')

        path = val.require_file(self.path, 'file_path', missing_ok=False)

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

        normalized: schema.ManifestEntry = {
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
        mapping = val.require_mapping(self.band_mapping, 'band_mapping')
        normalized: dict[int, str] = {}
        for k, v in mapping.items():
            try:
                k = int(k)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f'Band mapping key {k} must be an integer'
                ) from e
            _k = val.require_numeric_min(k, f'band_{k}_key', 1, int)
            _v = val.require_string(v, f'band_{k}_value')
            normalized[_k] = _v
        if set(normalized.keys()) != set(range(1, len(normalized) + 1)):
            raise ValueError(
                f'Band mapping keys should be contiguous integers from 1, '
                f'got: {sorted(normalized.keys())}'
            )

        return normalized

    def _normalize_category(self) -> schema.AllowedCategory:
        allowed = typing.get_args(schema.AllowedCategory)
        return typing.cast(
            schema.AllowedCategory,
            val.require_value_from(self.category, 'category', allowed, str)
        )

    def _normalize_categorical_specs(self) -> geo_core.CategoricalSpec:
        specs = val.require_mapping(self.cat_specs, 'specs')
        _specs = self._init_categorical_specs(specs)

        # add optional fields
        cls = specs.get('class_name')
        if cls is not None:
            _specs['class_name'] = val.require_mapping_str_v(cls, 'class_name')

        color_map = specs.get('color_map')
        if color_map is not None:
            _cmap = val.require_mapping(color_map, 'color_map')
            validated_cmp: dict[str, tuple[int, int, int]] = {}
            for k, rgb in _cmap.items():
                _rgb = val.require_numeric_sequence(rgb, f'RGB@{k}', int, 3)
                validated_cmp[k] = (_rgb[0], _rgb[1], _rgb[2])
            _specs['color_map'] = validated_cmp

        taxa = specs.get('taxonomy')
        if taxa is not None:
            if not cls:
                raise ValueError('Class names not provided for lookup')

            taxa = val.require_mapping(taxa, 'taxonomy')
            profile = val.require_string(taxa.get('profile'), 'profile')
            _specs['taxonomy'] = {'profile': profile}

        return _specs

    def _normalize_feature_schemes(
        self,
        band_mapping: dict[int, str]
    ) -> schema.FeatureSchemes:
        schemes = val.require_mapping(self.schemes, 'feature_schemes')

        valid_bands = list(band_mapping.values())
        normalized: dict[str, tuple[str, ...]] = {}
        for name, bands in schemes.items():
            val.require_sequence(bands, f'feature_scheme_{name}')
            b_seq = val.require_string_sequence(bands, f'scheme_{name}')
            for b in b_seq:
                val.require_value_from(b, f'scheme_{name}_band_{b}', valid_bands, str)
            normalized[name] = b_seq
        return normalized

    def _normalize_label_schemes(
        self,
        cat_specs: geo_core.CategoricalSpec
    ) -> schema.LabelSchemes:
        valid_range = range(
            cat_specs['index_base'],
            cat_specs['index_base'] + cat_specs['num_cls']
        )

        schemes = val.require_mapping(self.schemes, 'label_schemes')

        normalized: schema.LabelSchemes = {}
        for k, v in schemes.items():
            v = val.require_mapping(v, f'{k}_scheme')

            reclass = val.require_mapping(v.get('reclass'), 'reclass')
            norm_reclass: dict[str, tuple[int, ...]] = {}
            for grp_id, ids in reclass.items():
                ids_tuple = val.require_numeric_sequence(ids, 'class_IDs', int)
                norm_reclass[grp_id] = ids_tuple
                for cid in ids_tuple:
                    val.require_value_from(cid, 'class_IDs', valid_range, str)

            reclass_name = val.require_mapping(v.get('reclass_name'), 'reclass_name')
            valid_group_ids = list(reclass.keys())
            for grp_id in reclass_name.keys():
                val.require_value_from(grp_id, 'group_ID', valid_group_ids, str)

            normalized[k] = {
                'reclass': norm_reclass,
                'reclass_name': reclass_name,
            }

        return normalized

    @staticmethod
    def _init_categorical_specs(specs: dict) -> geo_core.CategoricalSpec:
        '''Initialize categorical specs with required fields.'''
        base = specs.get('index_base')
        num_cls = specs.get('num_cls')
        ignore = specs.get('ignore_cls')
        return {
            'index_base': val.require_numeric_min(base, 'index_base', 0, int),
            'num_cls': val.require_numeric_min(num_cls, 'num_cls', 1, int),
            'ignore_cls': val.require_numeric_sequence(ignore, 'ignore_cls', int)
        }
