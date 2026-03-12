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
Validation dataset scoring utilities that rank blocks by their label-
distribution similarity to a target. Computes target distributions from
global counts and scores blocks with a weighted L1 metric and reward
terms.

Public APIs:
    - BlockScore: Dataclass representing a scored block and its metadata.
    - ScoreParams: Dataclass holding scoring hyperparameters.
    - score_blocks: Compute and persist per-block scores based on label
      distributions.
'''

# standard imports
import dataclasses
import math
import os
# third-party imports
import numpy
# local imports
import landseg.prep_raster.blockbuilder as blockbuilder
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BlockScore:
    '''Scored validation-block record with location and score.'''
    name: str                       # block ID (e.g., "col_XXXX_row_YYYY")
    path: str                       # file path to the block artifact (.npz)
    col: int                        # column index parsed from the block name
    row: int                        # row index parsed from the block name
    score: float                    # computed score (lower is closer/better)

@dataclasses.dataclass
class ScoreParams:
    '''Score configuration.'''
    head: str                       # focal label head name (e.g., "layer1")
    alpha: float                    # exponent for transforming counts
    beta: float                     # reward weight for classes
    eps: float                      # small constant for numerical stability
    reward_cls: tuple[int, ...]     # class indices to reward

# -------------------------------Public Function-------------------------------
def score_blocks(
    global_cls_count: dict[str, list[int]],
    input_blocks: dict[str, str],
    params: ScoreParams,
    scores_path: str,
    *,
    rescore: bool = False
) -> list[BlockScore]:
    '''
    Score input blocks based on label class distributions.

    Args:
        global_cls_count: Global per-class counts by head used to derive
            the target distribution.
        input_blocks: Mapping from block names to file paths to score.
        params: Scoring parameters (head/alpha/beta/eps/reward classes).
        scores_path: Output JSON path for persisted, sorted scores.
        rescore: If True, recompute even if scores_path already exists.

    Returns:
        list[BlockScore]: Sorted scores (ascending by score).
    '''

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

# ------------------------------private  function------------------------------
def _score(
    block: tuple[str, str],
    head: str,
    target_p: numpy.ndarray,
    params: ScoreParams
) -> BlockScore:
    '''Score a single block against the target distribution.'''

    # parse arguments
    name, fpath = block
    beta = params.beta
    eps = params.eps
    reward_cls = params.reward_cls

    # specified head label count read from the block
    cls_count = blockbuilder.DataBlock().load(fpath).meta['label_count'][head]

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
    '''Counts to smoothed distribution, optionally exponentiated.'''

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
    '''Weighted L1 distance with bonus on specified reward classes.'''

    # sanity check
    assert numpy.isclose(p.sum(), 1), p.sum()
    assert numpy.isclose(q.sum(), 1), q.sum()
    # p as target q as test
    # use log weight and bonus for reward classes > target classes
    w_l1 = sum(abs(a - b) * (1 + abs(math.log(a + eps))) for a, b in zip(p, q))
    bonus = sum(max(0, q[i] - p[i]) for i in reward_cls) * beta
    return float(w_l1 - bonus)
