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
from landseg.geopipe.harmonize.manifest.normalizer import ManifestEntryNormalizer


# ----- `ManifestEntryNormalizer` features tests
def test_normalize_features_entry_basic():
    '''
    Given: A valid features manifest entry dictionary.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Return canonical entry with None categorical specs and schemes.
    '''
    raw_entry = {
        'name': 's2_sample',
        'path': '/path/to/s2.tif',
        'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
        'category': 'features',
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    assert entry['name'] == 's2_sample'
    assert entry['path'] == str(pathlib.Path(raw_entry['path']))
    assert entry['category'] == 'features'
    assert entry['band_mapping'] == {1: 'blue', 2: 'green', 3: 'red'}
    assert entry['categorical_specs'] is None
    assert entry['schemes'] is None


def test_normalize_features_entry_with_schemes():
    '''
    Given: A features entry with valid named feature schemes.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Return canonical entry preserving valid feature schemes.
    '''
    raw_entry = {
        'name': 's2_sample',
        'path': '/path/to/s2.tif',
        'band_mapping': {1: 'blue', 2: 'green', 3: 'red', 4: 'nir'},
        'category': 'features',
        'schemes': {
            'rgb': ['blue', 'green', 'red'],
            'rgb_nir': ['blue', 'green', 'red', 'nir'],
        },
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    assert entry['schemes'] == raw_entry['schemes']


def test_normalize_features_entry_invalid_scheme_band():
    '''
    Given: A features entry with scheme band name not in band mapping.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating missing band.
    '''
    raw_entry = {
        'name': 's2_sample',
        'path': '/path/to/s2.tif',
        'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
        'category': 'features',
        'schemes': {
            'rgb_swir': ['blue', 'green', 'swir1'],
        },
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='not in band_mapping'):
        _ = normalizer.normalized_entry


def test_normalize_features_entry_empty_scheme():
    '''
    Given: A features entry with an empty scheme band list.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating non-empty list requirement.
    '''
    raw_entry = {
        'name': 's2_sample',
        'path': '/path/to/s2.tif',
        'band_mapping': {1: 'blue', 2: 'green'},
        'category': 'features',
        'schemes': {'empty': []},
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='non-empty list'):
        _ = normalizer.normalized_entry


# ----- `ManifestEntryNormalizer` labels tests
def test_normalize_labels_entry_basic():
    '''
    Given: A valid label entry with required categorical specs.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Return canonical entry with normalized categorical specs.
    '''
    raw_entry = {
        'name': 'landcover',
        'path': '/path/to/lc.tif',
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
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    cat_specs = entry['categorical_specs']
    assert cat_specs is not None
    assert cat_specs['index_base'] == 1
    assert cat_specs['num_cls'] == 3
    assert cat_specs['ignore_cls'] == [255]
    assert cat_specs.get('class_name') == {
        '1': 'conifer',
        '2': 'decid',
        '3': 'water',
    }


def test_normalize_labels_entry_with_schemes():
    '''
    Given: A label entry with valid hierarchical label schemes.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Return canonical entry with validated label schemes.
    '''
    raw_entry = {
        'name': 'landcover',
        'path': '/path/to/lc.tif',
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
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    assert entry['schemes'] == raw_entry['schemes']


def test_normalize_labels_entry_scheme_class_out_of_range():
    '''
    Given: A label scheme with class ID outside valid index range.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating class ID is outside range.
    '''
    raw_entry = {
        'name': 'landcover',
        'path': '/path/to/lc.tif',
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
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='outside valid class range'):
        _ = normalizer.normalized_entry


def test_normalize_labels_entry_scheme_reclass_name_mismatch():
    '''
    Given: A label scheme with reclass_name group not in reclass.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating group mismatch.
    '''
    raw_entry = {
        'name': 'landcover',
        'path': '/path/to/lc.tif',
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
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='does not exist in reclass'):
        _ = normalizer.normalized_entry


# ----- `ManifestEntryNormalizer` domains tests
def test_normalize_domains_entry():
    '''
    Given: A valid domain entry with categorical specs.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Return canonical domain entry with categorical specs.
    '''
    raw_entry = {
        'name': 'ecodistrict',
        'path': '/path/to/eco.tif',
        'band_mapping': {1: 'ecodistrict'},
        'category': 'domains',
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 12,
            'ignore_cls': [],
        },
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    assert entry['category'] == 'domains'
    cat_specs = entry['categorical_specs']
    assert cat_specs is not None
    assert cat_specs['num_cls'] == 12


def test_normalize_domains_entry_with_schemes_raises():
    '''
    Given: A domain entry defining schemes.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating domain cannot define schemes.
    '''
    raw_entry = {
        'name': 'ecodistrict',
        'path': '/path/to/eco.tif',
        'band_mapping': {1: 'ecodistrict'},
        'category': 'domains',
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 5,
            'ignore_cls': [],
        },
        'schemes': {'dummy': ['eco']},
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='should not define "schemes"'):
        _ = normalizer.normalized_entry


def test_normalize_band_mapping_string_keys():
    '''
    Given: A band mapping with string integer keys from JSON.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Convert keys to integer keys contiguous from 1.
    '''
    raw_entry = {
        'name': 's2_sample',
        'path': '/path/to/s2.tif',
        'band_mapping': {'1': 'blue', '2': 'green'},
        'category': 'features',
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    assert entry['band_mapping'] == {1: 'blue', 2: 'green'}


# ----- `ManifestEntryNormalizer` taxonomy tests
def test_normalize_categorical_specs_with_taxonomy():
    '''
    Given: A label entry with valid taxonomy profile and class names.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Correctly validate and attach resolved taxonomy specs.
    '''
    raw_entry = {
        'name': 'species',
        'path': '/path/to/spc.tif',
        'band_mapping': {1: 'species'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'class_name': {'1': 'SB_BLACK_SPRUCE', '2': 'PJ_JACK_PINE'},
            'taxonomy': {
                'profile': 'ontario_tree_species_grouped_profiles',
            },
        },
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    entry = normalizer.normalized_entry
    cat_specs = entry['categorical_specs']
    assert cat_specs is not None
    taxa = cat_specs.get('taxonomy')
    assert taxa is not None
    assert taxa['profile'] == 'ontario_tree_species_grouped_profiles'
    canonical_indices = taxa.get('canonical_indices')
    assert canonical_indices is not None
    assert canonical_indices['1'] == 0
    assert canonical_indices['2'] == 3


def test_normalize_categorical_specs_taxonomy_missing_class_name():
    '''
    Given: Taxonomy declared without class_name provided.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating class names required.
    '''
    raw_entry = {
        'name': 'species',
        'path': '/path/to/spc.tif',
        'band_mapping': {1: 'species'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'taxonomy': {
                'profile': 'ontario_tree_species_grouped_profiles',
            },
        },
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='Class names not provided'):
        _ = normalizer.normalized_entry


# ----- `ManifestEntryNormalizer` validation error tests
def test_normalize_invalid_category():
    '''
    Given: An entry with an unsupported category name.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating category must be allowed.
    '''
    raw_entry = {
        'name': 'sample',
        'path': '/path/to/sample.tif',
        'band_mapping': {1: 'band1'},
        'category': 'unsupported_cat',
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='Invalid category'):
        _ = normalizer.normalized_entry


def test_normalize_band_mapping_non_contiguous():
    '''
    Given: A band mapping with non-contiguous keys starting from 1.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating non-contiguous integer keys.
    '''
    raw_entry = {
        'name': 'sample',
        'path': '/path/to/sample.tif',
        'band_mapping': {1: 'blue', 3: 'red'},
        'category': 'features',
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    with pytest.raises(ValueError, match='contiguous integers from 1'):
        _ = normalizer.normalized_entry


def test_normalize_categorical_specs_invalid_bounds():
    '''
    Given: Categorical specs with negative index_base and 0 num_cls.
    When: `ManifestEntryNormalizer.normalized_entry` is evaluated.
    Then: Raise ValueError indicating min bound violation.
    '''
    # negative index_base
    raw_entry_neg = {
        'name': 'lc',
        'path': '/path/to/lc.tif',
        'band_mapping': {1: 'lc'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': -1,
            'num_cls': 2,
            'ignore_cls': [],
        },
    }
    with pytest.raises(ValueError, match='< min value 0'):
        _ = ManifestEntryNormalizer(raw_entry_neg).normalized_entry

    # 0 num_cls
    raw_entry_zero = {
        'name': 'lc',
        'path': '/path/to/lc.tif',
        'band_mapping': {1: 'lc'},
        'category': 'labels',
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 0,
            'ignore_cls': [],
        },
    }
    with pytest.raises(ValueError, match='< min value 1'):
        _ = ManifestEntryNormalizer(raw_entry_zero).normalized_entry


def test_normalize_input_types():
    '''
    Given: Non-dictionary or empty dictionary inputs.
    When: Initializing or evaluating `ManifestEntryNormalizer`.
    Then: Raise appropriate TypeError or ValueError.
    '''
    with pytest.raises(TypeError, match='must be a dict'):
        ManifestEntryNormalizer('not_a_dict')

    with pytest.raises(ValueError, match='Input dict is empty'):
        ManifestEntryNormalizer({})


def test_normalizer_validate_method():
    '''
    Given: A valid manifest entry.
    When: Calling `validate()`.
    Then: Complete successfully without raising.
    '''
    raw_entry = {
        'name': 's2',
        'path': '/path/to/s2.tif',
        'band_mapping': {1: 'red'},
        'category': 'features',
    }
    normalizer = ManifestEntryNormalizer(raw_entry)
    normalizer.validate()
