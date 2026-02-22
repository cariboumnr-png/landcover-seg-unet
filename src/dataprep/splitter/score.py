'''Select validation dataset pipeline.'''

# standard imports
import dataclasses
import math
import os
# third-party imports
import numpy
# local imports
import dataprep
import utils

#
@dataclasses.dataclass
class BlockScore:
    '''doc'''
    name: str
    path: str
    col: int
    row: int
    score: float

@dataclasses.dataclass
class ScoreParams:
    '''Score configuration.'''
    head: str
    alpha: float
    beta: float
    eps: float
    reward_cls: tuple[int, ...]

#
def score_blocks(
    global_cls_count: dict[str, list[int]],
    input_blocks: dict[str, str],
    params: ScoreParams,
    scores_path: str,
    *,
    rescore: bool = False
) -> list[BlockScore]:
    '''Score inputs blocks based on its label class distribution.'''

    # read saved scores if already exist
    if os.path.exists(scores_path) and not rescore:
        return [BlockScore(**s) for s in utils.load_json(scores_path)]

    # target inverse probability from global class distribution
    head = params.head
    tgt = _count_to_inv_prob(global_cls_count[head], params.alpha, params.eps)
    # score blocks from list with parallel processing
    jobs = [(_score, (b, head, tgt, params), {}) for b in input_blocks.items()]
    scores = utils.ParallelExecutor().run(jobs)
    sorted_scores = sorted(scores, key=lambda _: _.score)

    # save sorted scores to a file
    utils.write_json(scores_path, [dataclasses.asdict(s) for s in sorted_scores])
    utils.hash_artifacts(scores_path)
    return sorted_scores

def _score(
    block: tuple[str, str],
    head: str,
    target_p: numpy.ndarray,
    params: ScoreParams
) -> BlockScore:
    '''Score each block.'''

    # parse arguments
    name, fpath = block
    beta = params.beta
    eps = params.eps
    reward_cls = params.reward_cls

    # specified head label count read from the block
    cls_count = dataprep.DataBlock().load(fpath).meta['label_count'][head]

    # split name string and get coords
    split = name.split('_')
    col = int(split[1])
    row = int(split[3])

    # block class distributions
    block_p = _count_to_inv_prob(cls_count, alpha=1.0, eps=eps)
    # L1 distance to measure pairwise closeness
    _score = _weighted_l1_w_reward(target_p, block_p, reward_cls, beta, eps)

    # return
    return BlockScore(name, fpath, col, row, _score)

def _count_to_inv_prob(
    cls_counts: list[int],
    alpha: float,
    eps: float
) -> numpy.ndarray:
    '''Global count to inverse distribution with epsilon smoothing.'''

    # safe count to probability distribution
    arr = numpy.asarray(cls_counts)
    tt = arr.sum()
    assert tt > 0
    p = arr / tt
    p = numpy.clip(p, eps, 1.0) # avoid downstream log(0)
    p = p / p.sum() # re-normalize to have probability sum==1

    # inverse and return
    inv = p ** (alpha)
    return inv / inv.sum()

def _weighted_l1_w_reward(
    p: numpy.ndarray,
    q: numpy.ndarray,
    reward_cls: tuple[int, ...],
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
