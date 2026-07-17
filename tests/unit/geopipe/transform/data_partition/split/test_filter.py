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

'''Unit tests for tile coordinate filtering logic (filter.py).'''

# local imports
import landseg.geopipe.transform.data_partition.split.filter as filter_mod


# ----- `filter_safe_tiles` tests
def test_filter_safe_tiles_overlap():
    '''
    Given: candidate tiles that overlap with holdout base tiles.
    When: Running filter_safe_tiles.
    Then: Filter out any overlapping candidate blocks.
    '''
    candidates = [
        (0, 0),
        (0, 50),
        (500, 500),
    ]
    base_tiles = [(0, 0)]

    result = filter_mod.filter_safe_tiles(
        candidates,
        base_tiles,
        block_size=256,
        block_stride=128,
        buffer_steps=0
    )

    assert result == [(500, 500)]


def test_filter_safe_tiles_buffer():
    '''
    Given: Candidate tiles near base tiles.
    When: Running filter_safe_tiles with a buffer_step > 0.
    Then: Exclude candidates falling within the spatial buffer zone.
    '''
    candidates = [(300, 0), (600, 0)]
    base_tiles = [(0, 0)]

    result = filter_mod.filter_safe_tiles(
        candidates,
        base_tiles,
        block_size=256,
        block_stride=128,
        buffer_steps=1
    )

    assert result == [(600, 0)]
