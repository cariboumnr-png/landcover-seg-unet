# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for domain mapping module (foundation_domain_map.py).'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.core.foundation_domain_map as domain_map


# ----- `DomainTileMap` tests
def test_domain_tile_map_container():
    '''
    Given: An empty initialized `DomainTileMap`.
    When: Setting data elements and testing container protocol.
    Then: Correctly stores entries, validates indices, and tracks length.
    '''
    tile_map = domain_map.DomainTileMap()
    assert len(tile_map) == 0

    # set data manually
    tile_map._data[(0, 0)] = {
        'majority': 1,
        'major_freq': 0.8,
        'pca_feature': [0.1, 0.2]
    }

    assert len(tile_map) == 1
    assert list(tile_map.keys()) == [(0, 0)]
    assert tile_map[(0, 0)]['majority'] == 1

    # index check
    with pytest.raises(TypeError, match='Index must be \\(x, y\\) in pixels'):
        _ = tile_map['invalid'] # type: ignore


def test_domain_tile_map_from_dict():
    '''
    Given: An in-memory dict of domain tiles.
    When: Constructed via `from_dict`.
    Then: Restore values and correctly expose descriptive properties.
    '''
    _tile: domain_map.DomainTile = {
        'majority': 2,
        'major_freq': 0.9,
        'pca_feature': [0.5]
    }
    tiles = {(10, 20): _tile}
    tile_map = domain_map.DomainTileMap.from_dict(tiles)

    assert len(tile_map) == 1
    assert tile_map[(10, 20)]['majority'] == 2

    # verify properties
    tile_map.meta['max_index'] = 5
    tile_map.meta['pca_axes_n'] = 1
    assert tile_map.max_id == 5
    assert tile_map.n_pca_ax == 1


def test_domain_tile_map_serialization_roundtrip():
    '''
    Given: A populated `DomainTileMap`.
    When: Exported and restored via JSON payloads.
    Then: State, metadata, and string-to-tuple coordinates are fully preserved.
    '''
    _tile: domain_map.DomainTile = {
        'majority': 1,
        'major_freq': 0.8,
        'pca_feature': [0.1, 0.2]
    }
    tiles = {(0, 0): _tile}
    tile_map = domain_map.DomainTileMap.from_dict(tiles)

    # populate meta to satisfy serialization assertions
    tile_map.meta = {
        'world_grid_ids': ['test_grid'],
        'valid_threshold': 0.9,
        'target_variance': 0.95,
        'max_index': 2,
        'major_freq_mean': 0.8,
        'major_freq_min': 0.7,
        'pca_axes_n': 2,
        'explained_variance': 0.96,
    }

    payload = tile_map.to_json_payload()

    # check string coordinate serialization
    assert '0,0' in payload['data']
    assert payload['data']['0,0']['majority'] == 1

    # restore
    restored = domain_map.DomainTileMap.from_json_payload(payload)

    assert len(restored) == 1
    assert (0, 0) in restored
    assert restored[(0, 0)]['majority'] == 1
    assert restored.meta['valid_threshold'] == 0.9
