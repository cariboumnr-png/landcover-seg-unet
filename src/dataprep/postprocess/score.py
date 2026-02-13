'''Select validation dataset pipeline.'''

# standard imports
import math
import os
import re
import typing
# third-party imports
import numpy
# local imports
import dataprep
import utils

# Public Type
class BlockScore(typing.TypedDict):
    '''dco'''
    name: str
    path: str
    col: int
    row: int
    score: float

# score all valid blocks based on label class distribution
def score(
    lbl_count_fpath: str,
    valid_blks_fpath: str,
    blks_scores_fpath: str,
    cache_cfg: utils.ConfigAccess,
    logger: utils.Logger,
) -> list[BlockScore]:
    '''doc'''

    # get a child logger
    logger = logger.get_child('score')

    # get scoring paramters
    score_param = cache_cfg.get_section_as_dict('scoring')

    # get overwrite option
    overwrite = cache_cfg.get_option('flags', 'overwrite_scores')

    # check if already scored
    if os.path.exists(blks_scores_fpath) and not overwrite:
        logger.log('INFO', f'Gathering block scoring from: {blks_scores_fpath}')
        return utils.load_json(blks_scores_fpath)

    # load valid block list
    blks: dict[str, str] = utils.load_json(valid_blks_fpath)
    global_count: dict[str, list[int]] = utils.load_json(lbl_count_fpath)

    # score blocks from list with parallel processing
    target_p  = _count_to_inv_prob(global_count, **score_param)
    jobs = [(_score_block, (b, target_p,), score_param) for b in blks.values()]
    scores = utils.ParallelExecutor().run(jobs)
    sorted_scores = sorted(scores, key=lambda _: _['score'])

    # save sorted scores to a file
    utils.write_json(blks_scores_fpath, sorted_scores)
    return sorted_scores

def _score_block(
    block_fpath: str,
    target_p: numpy.ndarray,
    **kwargs
) -> BlockScore:
    '''Score each block.'''

    # read from block
    meta = dataprep.DataBlock().load(block_fpath).meta
    name = meta['block_name']

    # find pattern from string
    pattern = r'col_(\d+)_row_(\d+)'
    matched = re.search(pattern, name)
    # there should be just one match
    if not matched:
        raise ValueError(f'Block naming pattern {pattern} not found')
    # get col and row
    col = int(matched.group(1))
    row = int(matched.group(2))

    # parse from parameter dict
    b = kwargs.get('beta', 1.0)
    e = kwargs.get('epsilon', 1e-12)
    reward_cls = kwargs.get('reward', [])

    # block class distributions
    block_p = _count_to_inv_prob(meta['label_count'], alpha=1.0)
    # L1 distance to measure pairwise closeness
    _score = _weighted_l1_w_reward(target_p, block_p, reward_cls, b, e)

    # return
    output: BlockScore = {
        'name': name,
        'path': block_fpath,
        'col': col,
        'row': row,
        'score': _score,
        # 'block_p': [round(p, 4) for p in block_p],
        # 'off_target': [round(p, 4) for p in block_p - target_p]
    }
    return output

def _count_to_inv_prob(
    counts: dict[str, list[int]],
    **kwargs
) -> numpy.ndarray:
    '''Global count to inverse distribution with epsilon smoothing.'''

    # get parameters from kwargs
    layer = kwargs.get('layer', 'layer1')
    alpha = kwargs.get('alpha', 0.6)
    epsilon = kwargs.get('epsilon', 1e-12)

    # safe count to probability distribution
    arr = numpy.asarray(counts[layer])
    tt = arr.sum()
    assert tt > 0
    p = arr / tt
    p = numpy.clip(p, epsilon, 1.0) # avoid downstream log(0)
    p = p / p.sum() # re-normalize to have probability sum==1

    # inverse and return
    inv = p ** (alpha)
    return inv / inv.sum()

def _weighted_l1_w_reward(
    p: numpy.ndarray,
    q: numpy.ndarray,
    reward_cls: list[int],
    beta: float,
    eps: float
) -> float:
    '''Weighted L1 distance between distributions with rewards.'''

    # sanity check
    assert numpy.isclose(p.sum(), 1), p.sum()
    assert numpy.isclose(q.sum(), 1), q.sum()
    # p as target q as test
    # use log weight and bonus for reward classes > target classes
    w_l1 = sum(abs(a - b) * (1 + abs(math.log(a + eps))) for a, b in zip(p, q))
    bonus = sum(max(0, q[i] - p[i]) for i in reward_cls) * beta
    return float(w_l1 - bonus)
