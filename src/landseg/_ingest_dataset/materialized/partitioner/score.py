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
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class ScoreParams:
    '''Score configuration.'''
    alpha: float                    # exponent for transforming counts
    beta: float                     # reward weight for classes
    epsilon: float                  # small constant for numerical stability
    reward: tuple[int, ...]         # class indices to reward

# -------------------------------Public Function-------------------------------
def score_blocks(
    class_counts: list[int],
    input_blocks: dict[str, list[int]],
    params: ScoreParams,
    scores_path: str,
    *,
    rescore: bool = False
) -> None:
    '''
    Score input blocks based on label class distributions.

    Args:
        global_cls_count: Global per-class counts by head used to derive
            the target distribution.
        input_blocks: Mapping from block names to file paths to score.
        params: Scoring parameters (head/alpha/beta/eps/reward classes).
        scores_path: Output JSON path for persisted, sorted scores.
        rescore: If True, recompute even if scores_path already exists.
    '''

    # read saved scores if already exist
    if os.path.exists(scores_path) and not rescore:
        return

    # target inverse probability from global class distribution
    tgt = _count_to_inv_prob(class_counts, params.alpha, params.epsilon)
    # score blocks from list with parallel processing
    jobs = [(_score, (b, tgt, params), {}) for b in input_blocks.items()]
    scores: list[tuple[str, float]] = utils.ParallelExecutor().run(jobs)
    sorted_scores = sorted(scores, key=lambda x: x[1])

    # save sorted scores to a file
    utils.write_json(scores_path, dict(sorted_scores))
    utils.hash_artifacts(scores_path)

# ------------------------------private  function------------------------------
def _score(
    block: tuple[str, list[int]],
    target_p: numpy.ndarray,
    params: ScoreParams
) -> tuple[str, float]:
    '''Score a single block against the target distribution.'''

    # parse arguments
    name, cls_count = block
    beta = params.beta
    eps = params.epsilon
    reward_cls = params.reward

    # block class distributions
    block_p = _count_to_inv_prob(cls_count, alpha=1.0, eps=eps)
    # L1 distance to measure pairwise closeness
    _score = _weighted_l1_w_reward(target_p, block_p, reward_cls, beta, eps)

    # return
    return name, _score

def _count_to_inv_prob(
    cls_counts: list[int],
    alpha: float,
    eps: float
) -> numpy.ndarray:
    '''Counts to smoothed distribution, optionally exponentiated.'''

    # safe count to probability distribution
    arr = numpy.asarray(cls_counts)
    tt = arr.sum()
    if tt == 0:
        tt = 1
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
