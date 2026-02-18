'''Select validation dataset pipeline.'''

# standard imports
import math
# local imports
import dataprep
import utils

# split training blocks into training and validation dataset
def split_blocks(
    scores: list[dataprep.BlockScore],
    training_blocks: str,
    validation_blocks: str,
    logger: utils.Logger,
) -> None:
    '''Split validation/training data.'''

    # get validation blocks
    v_blks = _select_validation_blocks(scores, logger)
    t_blks = {b['name']: b['path'] for b in scores if b['name'] not in v_blks}

    # write validation blocks to json
    utils.write_json(validation_blocks, v_blks)
    # write training blocks to json
    utils.write_json(training_blocks, t_blks)

def _select_validation_blocks(
    scores: list[dataprep.BlockScore],
    logger: utils.Logger,
    **kwargs
) -> dict[str, str]:
    '''Select a set of validation blocks.'''

    valblk_per = kwargs.get('toprange', 0.1)
    min_dist = kwargs.get('mindist', 400)

    # get an expanded top ranking blocks
    num_blk = round(len(scores) * valblk_per)
    existing_locs = [(0, 0)] # init the list
    return_dict = {}

    # iterat through blocks
    for i, blk in enumerate(scores):
        x = blk['col']
        y = blk['row']
        assert isinstance(x, int) and isinstance(y, int) # typing sanity
        dd = _min_distance_from_locs(existing_locs, (x, y))
        if dd >= min_dist:
            existing_locs.append((x, y))
            return_dict[blk['name']] = blk['path']
        if len(return_dict) == num_blk:
            logger.log('INFO', f'Gathered enough blocks at block {i + 1}')
            break
    logger.log('INFO', f'Gathered {len(return_dict)} blocks from all')

    # return
    return return_dict

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
