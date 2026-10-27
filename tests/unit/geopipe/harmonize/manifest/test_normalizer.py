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

'''Unit tests for dataset manifest normalizer (normalizer.py).'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.geopipe.harmonize.manifest.normalizer as normalizer_mod
import landseg.utils.validator as val


normalize_manifest_entry = normalizer_mod.normalize_manifest_entry


# ----- `normalize_manifest_entry` features tests
def test_normalize_features_entry_basic(tmp_path):
    '''
    Given: A valid features manifest entry dictionary.
    When: `normalize_manifest_entry` is evaluated.
    Then: Return canonical entry with None categorical specs and schemes.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 's2_sample',
        'path': sample_file,
        'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
        'category': 'features',
    }
    entry = normalize_manifest_entry(raw_entry)
    assert entry['name'] == 's2_sample'
    assert entry['path'] == sample_file
    assert entry['category'] == 'features'
    assert entry['band_mapping'] == {1: 'blue', 2: 'green', 3: 'red'}
    assert entry['categorical_specs'] is None
    assert entry['schemes'] is None


def test_normalize_features_entry_with_schemes(tmp_path):
    '''
    Given: A features entry with valid named feature schemes.
    When: `normalize_manifest_entry` is evaluated.
    Then: Return canonical entry preserving valid feature schemes.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 's2_sample',
        'path': sample_file,
        'band_mapping': {1: 'blue', 2: 'green', 3: 'red', 4: 'nir'},
        'category': 'features',
        'schemes': {
            'rgb': ['blue', 'green', 'red'],
            'rgb_nir': ['blue', 'green', 'red', 'nir'],
        },
    }
    entry = normalize_manifest_entry(raw_entry)
    assert entry['schemes'] == {
        'rgb': ('blue', 'green', 'red'),
        'rgb_nir': ('blue', 'green', 'red', 'nir'),
    }


def test_normalize_features_entry_invalid_scheme_band(tmp_path):
    '''
    Given: A features entry with scheme band name not in band mapping.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValidationError indicating band not in allowed pool.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 's2_sample',
        'path': sample_file,
        'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
        'category': 'features',
        'schemes': {'rgb_swir': ['blue', 'green', 'swir1']},
    }
    with pytest.raises(val.ValidationError, match='not in range'):
        _ = normalize_manifest_entry(raw_entry)


def test_normalize_features_entry_empty_scheme(tmp_path):
    '''
    Given: A features entry with an empty scheme band list.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValueError indicating non-empty list requirement.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 's2_sample',
        'path': sample_file,
        'band_mapping': {1: 'blue', 2: 'green'},
        'category': 'features',
        'schemes': {'empty': []},
    }
    with pytest.raises(ValueError, match='non-empty list'):
        _ = normalize_manifest_entry(raw_entry)


# ----- `normalize_manifest_entry` labels tests
def test_normalize_labels_entry_basic(tmp_path):
    '''
    Given: A valid label entry with required categorical specs.
    When: `normalize_manifest_entry` is evaluated.
    Then: Return canonical entry with normalized categorical specs.
    '''
    sample_file = _create_sample_file(tmp_path, 'lc.tif')
    raw_entry = {
        'name': 'landcover',
        'path': sample_file,
        'band_mapping': {1: 'landcover'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 3,
            'ignore_cls': [255],
            'class_name': {'1': 'conifer', '2': 'decid', '3': 'water'},
            'color_map': {
                '1': [0, 100, 0],
                '2': [34, 139, 34],
                '3': [0, 0, 255],
            },
        },
    }
    entry = normalize_manifest_entry(raw_entry)
    cat_specs = entry['categorical_specs']
    assert cat_specs is not None
    assert cat_specs['index_base'] == 1
    assert cat_specs['num_cls'] == 3
    assert cat_specs['ignore_cls'] == (255,)
    assert cat_specs.get('color_map') == {
        '1': (0, 100, 0),
        '2': (34, 139, 34),
        '3': (0, 0, 255),
    }
    assert cat_specs.get('class_name') == {
        '1': 'conifer',
        '2': 'decid',
        '3': 'water',
    }


def test_normalize_labels_entry_with_schemes(tmp_path):
    '''
    Given: A label entry with valid hierarchical label schemes.
    When: `normalize_manifest_entry` is evaluated.
    Then: Return canonical entry with validated label schemes.
    '''
    sample_file = _create_sample_file(tmp_path, 'lc.tif')
    raw_entry = {
        'name': 'landcover',
        'path': sample_file,
        'band_mapping': {1: 'landcover'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 3,
            'ignore_cls': [255],
        },
        'schemes': {
            'binary': {
                'reclass': {'1': [1, 2], '2': [3]},
                'reclass_name': {'1': 'forest', '2': 'water'},
            },
        },
    }
    entry = normalize_manifest_entry(raw_entry)
    assert entry['schemes'] == {
        'binary': {
            'reclass': {'1': (1, 2), '2': (3,)},
            'reclass_name': {'1': 'forest', '2': 'water'},
        },
    }


def test_normalize_labels_entry_scheme_class_out_of_range(tmp_path):
    '''
    Given: A label scheme with class ID outside valid index range.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValidationError indicating class ID is outside range.
    '''
    sample_file = _create_sample_file(tmp_path, 'lc.tif')
    raw_entry = {
        'name': 'landcover',
        'path': sample_file,
        'band_mapping': {1: 'landcover'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
        },
        'schemes': {
            'binary': {
                'reclass': {'1': [1], '2': [3]}, # 3 is out of [1..2]
                'reclass_name': {'1': 'WAT', '2': 'VEG'},
            },
        },
    }
    with pytest.raises(val.ValidationError, match='> maximum 2'):
        _ = normalize_manifest_entry(raw_entry)


def test_normalize_labels_entry_scheme_reclass_name_mismatch(tmp_path):
    '''
    Given: A label scheme with reclass_name group not in reclass.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValidationError indicating group mismatch.
    '''
    sample_file = _create_sample_file(tmp_path, 'lc.tif')
    raw_entry = {
        'name': 'landcover',
        'path': sample_file,
        'band_mapping': {1: 'landcover'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
        },
        'schemes': {
            'binary': {
                'reclass': {'1': [1, 2]},
                'reclass_name': {'1': 'forest', '2': 'orphan'},
            },
        },
    }
    with pytest.raises(val.ValidationError, match='not in range'):
        _ = normalize_manifest_entry(raw_entry)


# ----- `normalize_manifest_entry` domains tests
def test_normalize_domains_entry(tmp_path):
    '''
    Given: A valid domain entry with categorical specs.
    When: `normalize_manifest_entry` is evaluated.
    Then: Return canonical domain entry with categorical specs.
    '''
    sample_file = _create_sample_file(tmp_path, 'eco.tif')
    raw_entry = {
        'name': 'ecodistrict',
        'path': sample_file,
        'band_mapping': {1: 'ecodistrict'},
        'category': 'domains',
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 12,
            'ignore_cls': [],
        },
    }
    entry = normalize_manifest_entry(raw_entry)
    assert entry['category'] == 'domains'
    cat_specs = entry['categorical_specs']
    assert cat_specs is not None
    assert cat_specs['num_cls'] == 12


def test_normalize_domains_entry_with_schemes_raises(tmp_path):
    '''
    Given: A domain entry defining schemes.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValueError indicating domain cannot define schemes.
    '''
    sample_file = _create_sample_file(tmp_path, 'eco.tif')
    raw_entry = {
        'name': 'ecodistrict',
        'path': sample_file,
        'band_mapping': {1: 'ecodistrict'},
        'category': 'domains',
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 5,
            'ignore_cls': [],
        },
        'schemes': {'dummy': ['eco']},
    }
    with pytest.raises(ValueError, match='should not define "schemes"'):
        _ = normalize_manifest_entry(raw_entry)


def test_normalize_band_mapping_string_keys(tmp_path):
    '''
    Given: A band mapping with string integer keys from JSON.
    When: `normalize_manifest_entry` is evaluated.
    Then: Convert keys to integer keys contiguous from 1.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 's2_sample',
        'path': sample_file,
        'band_mapping': {'1': 'blue', '2': 'green'},
        'category': 'features',
    }
    entry = normalize_manifest_entry(raw_entry)
    assert entry['band_mapping'] == {1: 'blue', 2: 'green'}


# ----- `normalize_manifest_entry` taxonomy tests
def test_normalize_categorical_specs_with_taxonomy(tmp_path):
    '''
    Given: A label entry with valid taxonomy profile and class names.
    When: `normalize_manifest_entry` is evaluated.
    Then: Correctly preserve declared taxonomy profile.
    '''
    sample_file = _create_sample_file(tmp_path, 'spc.tif')
    raw_entry = {
        'name': 'species',
        'path': sample_file,
        'band_mapping': {1: 'species'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'class_name': {'1': 'SB_BLACK_SPRUCE', '2': 'PJ_JACK_PINE'},
            'taxonomy': {'profile': 'ontario_tree_species_grouped_profiles'},
        },
    }
    entry = normalize_manifest_entry(raw_entry)
    cat_specs = entry['categorical_specs']
    assert cat_specs is not None
    taxa = cat_specs.get('taxonomy')
    assert taxa == {'profile': 'ontario_tree_species_grouped_profiles'}


def test_normalize_categorical_specs_taxonomy_missing_class_name(tmp_path):
    '''
    Given: Taxonomy declared without class_name provided.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValueError indicating class names required.
    '''
    sample_file = _create_sample_file(tmp_path, 'spc.tif')
    raw_entry = {
        'name': 'species',
        'path': sample_file,
        'band_mapping': {1: 'species'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'taxonomy': {'profile': 'ontario_tree_species_grouped_profiles'},
        },
    }
    with pytest.raises(ValueError, match='Class names not provided'):
        _ = normalize_manifest_entry(raw_entry)


# ----- `normalize_manifest_entry` validation error tests
def test_normalize_invalid_category(tmp_path):
    '''
    Given: An entry with an unsupported category name.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValidationError indicating category must be allowed.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 'sample',
        'path': sample_file,
        'band_mapping': {1: 'band1'},
        'category': 'unsupported_cat',
    }
    with pytest.raises(val.ValidationError, match='not in range'):
        _ = normalize_manifest_entry(raw_entry)


def test_normalize_band_mapping_non_contiguous(tmp_path):
    '''
    Given: A band mapping with non-contiguous keys starting from 1.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValueError indicating non-contiguous integer keys.
    '''
    sample_file = _create_sample_file(tmp_path)
    raw_entry = {
        'name': 'sample',
        'path': sample_file,
        'band_mapping': {1: 'blue', 3: 'red'},
        'category': 'features',
    }
    with pytest.raises(ValueError, match='contiguous integers from 1'):
        _ = normalize_manifest_entry(raw_entry)


def test_normalize_categorical_specs_invalid_bounds(tmp_path):
    '''
    Given: Categorical specs with negative index_base and 0 num_cls.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValidationError indicating min bound violation.
    '''
    sample_file = _create_sample_file(tmp_path, 'lc.tif')
    # negative index_base
    raw_entry_neg = {
        'name': 'lc',
        'path': sample_file,
        'band_mapping': {1: 'lc'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': -1,
            'num_cls': 2,
            'ignore_cls': [],
        },
    }
    with pytest.raises(val.ValidationError, match='< minimum 0'):
        _ = normalize_manifest_entry(raw_entry_neg)

    # 0 num_cls
    raw_entry_zero = {
        'name': 'lc',
        'path': sample_file,
        'band_mapping': {1: 'lc'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 0,
            'ignore_cls': [],
        },
    }
    with pytest.raises(val.ValidationError, match='< minimum 1'):
        _ = normalize_manifest_entry(raw_entry_zero)


def test_normalize_categorical_specs_invalid_color_map_bounds(tmp_path):
    '''
    Given: Categorical specs with RGB values exceeding 255.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise ValidationError indicating color channel > maximum 255.
    '''
    sample_file = _create_sample_file(tmp_path, 'lc.tif')
    raw_entry = {
        'name': 'lc',
        'path': sample_file,
        'band_mapping': {1: 'lc'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [],
            'color_map': {'1': [0, 100, 300]},
        },
    }
    with pytest.raises(val.ValidationError, match='> maximum 255'):
        _ = normalize_manifest_entry(raw_entry)


def test_normalize_input_types():
    '''
    Given: Non-dictionary or empty dictionary inputs.
    When: `normalize_manifest_entry` is evaluated.
    Then: Raise appropriate TypeError or ValueError.
    '''
    with pytest.raises(TypeError, match='not a mapping'):
        normalizer_mod.normalize_manifest_entry('not_a_dict')

    with pytest.raises(ValueError, match='empty dict'):
        normalizer_mod.normalize_manifest_entry({})


# ----- private helpers
def _create_sample_file(
    tmp_path: pathlib.Path,
    filename: str = 'sample.tif',
) -> pathlib.Path:
    '''Create an empty raster placeholder file.'''
    path = tmp_path / filename
    path.touch(exist_ok=True)
    return path
