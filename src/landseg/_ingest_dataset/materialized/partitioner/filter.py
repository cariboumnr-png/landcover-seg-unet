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

'''
Oversample training blocks for class rebalancing.

This module provides utilities to filter candidate tile coordinates so
that training patches do not overlap validation/test patches. A small
configurable buffer can be applied in units of the candidate stride to
reduce context leakage across splits.
'''

#
def filter_safe_tiles(
    candidates: list[tuple[int, int]],
    base_tiles: list[tuple[int, int]],
    *,
    block_size: int,
    stride: int,
    buffer_steps: int = 1
) -> list[tuple[int, int]]:
    '''
    Keep candidate tile coords that do not overlap any base tile.

    A tile is excluded if there exists a base tile whose top-left is
    within the per-axis threshold:
        abs(dx) < T and abs(dy) < T,
    where T = block_size + buffer_steps * stride.

    Assumptions:
      - base_tiles lie on a grid with stride == block_size.
      - candidate tiles have stride == stride (0 < stride ≤ block_size).
      - both grids share the same top-left origin.
      - tiles are axis-aligned and of size block_size * block_size.

    Args:
        candidates:
            A list of (x, y) coordinate tuples representing candidate
            tile top-left positions. These tiles may use any stride
            satisfying 0 < stride ≤ block_size.

        base_tiles:
            A list of (x, y) coordinate tuples for base tiles that
            must not be overlapped. These tiles are assumed to lie on
            a grid whose stride equals block_size.

        block_size:
            The tile size in pixels for both candidate and base tiles.
            Tiles are axis-aligned squares of size block_size ** 2.

        stride:
            The stride of the candidate tiles. Must satisfy
            0 < stride ≤ block_size. Used for computing the expanded
            non-overlap threshold when buffer_steps > 0.

        buffer_steps:
            Number of stride units to expand the exclusion threshold.
            A buffer of buffer_steps x stride pixels is added along
            both axes to prevent near-overlap or receptive-field
            leakage. Set to 0 for strict geometric non-overlap.
            Defaults to 1.

    Returns:
        A list of (x, y) coordinate tuples from `candidates` whose tiles
        do not overlap any tile in `base_tiles` under the configured
        threshold.

    '''

    # sanity checks
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    if stride <= 0 or stride > block_size:
        raise ValueError('candidate_stride must satisfy 0 < s ≤ block_size')
    if buffer_steps < 0:
        raise ValueError('buffer_steps must be non-negative')

    keep: list[tuple[int, int]] = []
    base_set = set(base_tiles)
    # iterate through all stridden tiles
    for c in candidates:
        # local search
        if _overlaps_w_base(c, base_set, block_size, stride, buffer_steps):
            continue
        keep.append(c)

    # return
    return keep

def _overlaps_w_base(
    coord: tuple[int, int],
    base_tiles: set[tuple[int, int]],
    block_size: int,
    stride: int,
    buffer_steps: int
) -> bool:
    '''True if candidate tile overlaps any base tile within radius.'''

    # base grid indices containing the candidate's top-left
    xc, yc = coord
    i = xc // block_size
    j = yc // block_size

    # effective per-axis overlap threshold and search radius in base indices
    thres = block_size + buffer_steps * stride
    radius = 1 + (thres - 1) // block_size

    # iteration search within radius
    for di in range(-radius, radius + 1):
        bx = (i + di) * block_size
        # fast reject on x
        if abs(xc - bx) >= thres:
            continue
        # continue on y
        for dj in range(-radius, radius + 1):
            by = (j + dj) * block_size
            if abs(yc - by) >= thres:
                continue
            # return true if overlap
            if (bx, by) in base_tiles:
                return True
    return False
