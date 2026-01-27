'''Select validation dataset pipeline.'''

# standard imports
import math
import os
# local imports
import utils

def select_validation_blocks(
        scores: list[dict[str, str | int ]] ,
        valselect_param: dict,
        logger: utils.Logger
    ) -> dict[str, str]:
    '''Select a set of validation blocks.'''

    valblk_per = valselect_param.get('toprange', 0.1)
    min_dist = valselect_param.get('mindist', 400)

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
            return_dict[blk['block_name']] = blk['file_path']
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

def run(
        scores_fpath: str,
        v_fpath: str,
        t_fpath: str,
        valselect_param: dict,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> None:
    '''Split validation/training data.'''

    # get a child logger
    logger=logger.get_child('split')

    # skip if not to overwrite and all files are present
    if os.path.exists(v_fpath) and os.path.exists(t_fpath) and not overwrite:
        logger.log('INFO', 'Keeping existing validation/training dataset split')
        return

    # get validation blocks
    scores: list[dict[str, str | int ]] = utils.funcs.load_json(scores_fpath)
    v_blks = select_validation_blocks(scores, valselect_param, logger=logger)
    t_blks = {
        b['block_name']: b['file_path']
        for b in scores if b['block_name'] not in v_blks
    }

    # pickle validation blocks to a file
    utils.write_json(v_fpath, v_blks)
    # pickle training blocks to a file
    utils.write_json(t_fpath, t_blks)
