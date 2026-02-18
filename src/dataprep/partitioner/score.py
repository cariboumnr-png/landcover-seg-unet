'''Select validation dataset pipeline.'''

# standard imports
import math
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
def score_blocks(
    label_count: dict[str, list[int]],
    valid_blks: dict[str, str],
    config: dataprep.ScoringConfig,
    scores_fpath: str,
    logger: utils.Logger,
) -> list[BlockScore]:
    '''doc'''

    # score blocks from list with parallel processing
    target  = _count_to_inv_prob(label_count, **config)
    jobs = [(_score_block, (b, target,), config) for b in valid_blks.items()]
    scores = utils.ParallelExecutor().run(jobs)
    sorted_scores = sorted(scores, key=lambda _: _['score'])

    # save sorted scores to a file
    logger.log('INFO', f'Scores saved to {scores_fpath}')
    utils.write_json(scores_fpath, sorted_scores)
    return sorted_scores

def _score_block(
    block: tuple[str, str],
    target_p: numpy.ndarray,
    **kwargs
) -> BlockScore:
    '''Score each block.'''

    # parse
    name, fpath = block

    # read from block
    meta = dataprep.DataBlock().load(fpath).meta

    # split name string and get coords
    split = name.split('_')
    col = int(split[1])
    row = int(split[3])

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
        'path': fpath,
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
