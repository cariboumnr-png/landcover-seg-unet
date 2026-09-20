# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Spatial buffer filtering for candidate data blocks.

Provides utilities to filter candidate tile coordinates so that training
patches do not overlap validation or test patches, with configurable
buffer steps to reduce spatial autocorrelation leakage.

Public APIs:
    - filter_safe_tiles: keep candidate tile coordinates safely.
'''


# ----- public functions
def filter_safe_tiles(
    candidates: list[tuple[int, int]],
    base_tiles: list[tuple[int, int]],
    *,
    block_size: int,
    block_stride: int,
    buffer_steps: int = 1,
) -> list[tuple[int, int]]:
    '''
    Keep candidate tile coords that do not overlap any base tile.

    A tile is excluded if there exists a base tile whose top-left is
    within the per-axis threshold:
        abs(dx) < T and abs(dy) < T,
    where T = block_size + buffer_steps * stride.

    Assumptions:
      - base_tiles lie on a grid with stride == block_size.
      - candidate tiles have stride == stride (0 < stride <= size).
      - both grids share the same top-left origin.
      - tiles are axis-aligned and of size block_size * block_size.

    Args:
        candidates:
            list of (x, y) coordinate tuples representing candidate
            tile top-left positions.
        base_tiles:
            list of (x, y) coordinate tuples for base tiles that
            must not be overlapped.
        block_size:
            the tile size in pixels for both candidate and base tiles.
        block_stride:
            the stride of candidate tiles (0 < stride <= block_size).
        buffer_steps:
            number of stride units to expand exclusion threshold.

    Returns:
        list[tuple[int, int]]:
            candidate coordinates that do not overlap any base tiles.
    '''
    # sanity checks
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    if block_stride < 0 or block_stride > block_size:
        raise ValueError('candidate_stride must satisfy 0 <= s <= block_size')
    if buffer_steps < 0:
        raise ValueError('buffer_steps must be non-negative')

    keep: list[tuple[int, int]] = []
    base_set = set(base_tiles)
    # iterate through all stridden tiles
    for c in candidates:
        # local search
        if _overlaps_w_base(
            c, base_set, block_size, block_stride, buffer_steps
        ):
            continue
        keep.append(c)

    # return
    return keep


# ----- private helpers
def _overlaps_w_base(
    coord: tuple[int, int],
    base_tiles: set[tuple[int, int]],
    block_size: int,
    block_stride: int,
    buffer_steps: int,
) -> bool:
    '''True if candidate tile overlaps any base tile within radius.'''
    # base grid indices containing the candidate's top-left
    xc, yc = coord
    i = xc // block_size
    j = yc // block_size

    # per-axis overlap threshold and search radius in base indices
    thres = block_size + buffer_steps * block_stride
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
