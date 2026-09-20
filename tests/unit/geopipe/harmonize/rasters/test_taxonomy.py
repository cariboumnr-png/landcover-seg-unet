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

# pylint: disable=protected-access

'''Unit tests for taxonomy resolver and gatekeeper validation.'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.harmonize.taxonomy.taxonomy as taxonomy


# ----- `get_available_profiles` tests
def test_get_available_profiles_finds_registered_profiles():
    '''
    Given: The canonical knowledge base directory.
    When: `get_available_profiles` is called.
    Then: Return list containing registered profiles.
    '''
    profiles = taxonomy.get_available_profiles('knowledge')
    assert 'ontario_tree_species_grouped_profiles' in profiles
    assert 'ontario_tree_species_profiles' in profiles


def test_get_available_profiles_nonexistent_dir(tmp_path):
    '''
    Given: A non-existent root path.
    When: `get_available_profiles` is called.
    Then: Return an empty list without raising.
    '''
    empty_root = str(tmp_path / 'nonexistent')
    profiles = taxonomy.get_available_profiles(empty_root)
    assert profiles == []


# ----- `resolve_taxonomy_metadata` tests
def test_resolve_taxonomy_metadata_success():
    '''
    Given: A registered profile name.
    When: `_resolve_taxonomy_metadata` is called.
    Then: Return parsed dictionary with classes and model info.
    '''
    code_lookup = taxonomy._resolve_taxonomy_metadata(
        'ontario_tree_species_grouped_profiles',
        root='knowledge',
    )
    assert 'SB_BLACK_SPRUCE' in code_lookup
    assert code_lookup['SB_BLACK_SPRUCE']['code'] == 'SB_BLACK_SPRUCE'


def test_resolve_taxonomy_metadata_not_found(tmp_path):
    '''
    Given: An invalid profile name.
    When: `_resolve_taxonomy_metadata` is called.
    Then: Raise ValueError detailing available profiles.
    '''
    with pytest.raises(ValueError, match='Taxonomy profile "invalid_prof"'):
        taxonomy._resolve_taxonomy_metadata(
            'invalid_prof',
            root=str(tmp_path),
        )


# ----- `validate_specs` tests
def test_validate_specs_species_mapping():
    '''
    Given: A valid taxonomy spec with explicit species_mapping.
    When: `validate_specs` is called.
    Then: Return canonical taxonomy specs with matrix indices.
    '''
    species_mapping = {
        '1': 'SB_BLACK_SPRUCE',
        '2': 'PJ_JACK_PINE',
    }
    specs = taxonomy.validate_specs(
        'ontario_tree_species_grouped_profiles',
        species_mapping,
        num_cls=2,
        knowledge_root='knowledge',
    )
    assert specs['profile'] == 'ontario_tree_species_grouped_profiles'
    canonical_indices = specs.get('canonical_indices')
    assert canonical_indices is not None
    assert canonical_indices['1'] == 0
    assert canonical_indices['2'] == 3


def test_validate_specs_unknown_code():
    '''
    Given: A species code with an unknown code.
    When: `validate_specs` is called.
    Then: Raise ValueError.
    '''
    species_mapping = {
        '1': 'SBB',
        '2': 'PJ_JACK_PINE',
    }
    with pytest.raises(ValueError, match='Unknown taxonomy class code "SBB"'):
        taxonomy.validate_specs(
            'ontario_tree_species_grouped_profiles',
            species_mapping,
            num_cls=2,
            knowledge_root='knowledge',
        )


def test_validate_specs_count_mismatch():
    '''
    Given: Species mapping length that does not match num_cls.
    When: `validate_specs` is called.
    Then: Raise ValueError with count mismatch explanation.
    '''
    species_mapping = {
        '1': 'SB_BLACK_SPRUCE',
    }
    with pytest.raises(ValueError, match='Taxonomy declares 1 classes'):
        taxonomy.validate_specs(
            'ontario_tree_species_grouped_profiles',
            species_mapping,
            num_cls=2,
            knowledge_root='knowledge',
        )
