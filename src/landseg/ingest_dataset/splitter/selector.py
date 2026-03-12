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
Validation-block selection for choosing a geographically diverse subset
of scored blocks based on ranking and spatial distance constraints.

Public APIs:
    - select_val_blocks: Select validation blocks (score ranked) using
      a minimum-distance buffer.
'''

# standard imports
import math
# local imports
import landseg.ingest_dataset.splitter as splitter
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def select_val_blocks(
    scores: list[splitter.BlockScore],
    logger: utils.Logger,
    *,
    toprange: float = 0.1,
    min_distance: int = 400
) -> dict[str, str]:
    '''
    Select validation blocks based on ranking and spatial distance.

    Args:
        scores: Ranked list of BlockScore objects (ascending by score).
        logger: Logger for progress reporting.
        toprange: Fraction of top-ranked blocks eligible for selection.
            Defaults to 0.1.
        min_distance: Minimum Euclidean distance (grid units) required
            between any two selected blocks. Defaults to 400.

    Returns:
        dict: Mapping from block names to file paths for the
            selected validation set.
    '''

    # get an expanded top ranking blocks
    num_blk = round(len(scores) * toprange)
    existing_locs = [(0, 0)] # init the list
    return_dict = {}

    # iterat through blocks
    for i, blk in enumerate(scores):
        x = blk.col
        y = blk.row
        assert isinstance(x, int) and isinstance(y, int) # typing sanity
        dd = _min_distance_from_locs(existing_locs, (x, y))
        if dd >= min_distance:
            existing_locs.append((x, y))
            return_dict[blk.name] = blk.path
        if len(return_dict) == num_blk:
            logger.log('INFO', f'Gathered enough blocks at block {i + 1}')
            break
    logger.log('INFO', f'Gathered {len(return_dict)} blocks from all')

    # return
    return return_dict

# ------------------------------private  function------------------------------
def _min_distance_from_locs(
    existing_locs: list[tuple[int, int]],
    input_locs: tuple[int, int]
) -> float:
    '''Min Euclidean distance between input and existing locations.'''

    distances = []
    xx, yy = input_locs
    for x, y in existing_locs:
        distances.append(math.sqrt((xx - x) ** 2 + (yy - y) ** 2))
    if 0 in distances:
        distances.remove(0)
    return min(distances)
