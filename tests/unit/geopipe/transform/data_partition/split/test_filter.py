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

'''Unit tests for tile coordinate filtering logic (filter.py).'''

# local imports
import landseg.geopipe.transform.data_partition.split.filter as filter_mod


# ----- `filter_safe_tiles` tests
def test_filter_safe_tiles_overlap():
    # candidates list
    candidates = [
        (0, 0),
        (0, 50),  # overlaps (0,0) if size is 256
        (500, 500), # safe block
    ]
    base_tiles = [(0, 0)] # e.g. validation block

    result = filter_mod.filter_safe_tiles(
        candidates,
        base_tiles,
        block_size=256,
        block_stride=128,
        buffer_steps=0
    )

    # (0,0) and (0,50) overlap (0,0) with size 256. (500, 500) is far away.
    assert result == [(500, 500)]


def test_filter_safe_tiles_buffer():
    # base tile at (0, 0)
    # candidate at (300, 0) -> distance is 300
    # block size is 256.
    # if buffer_steps = 1 and block_stride = 128:
    # threshold = block_size + buffer_steps * block_stride = 256 + 128 = 384.
    # distance (300) < threshold (384), so it should be filtered out.
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
