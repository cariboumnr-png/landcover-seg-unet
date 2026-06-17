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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Aggregator for multi-task learning (MTL) metrics across multiple heads.

Provides horizontal evaluation of model performance across independent
and hierarchical tasks, including Global Exact Match (GEM) and
logical constraint violation detection.
'''

# standard imports
import dataclasses
# third-party imports
import torch
# local imports
import landseg.session.engine.runtime.tasks.constraints as constraints

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _Tally:
    '''Internal counter for horizontal metrics.'''
    hits: int = 0
    samples: int = 0

# --------------------------------Public  Class--------------------------------
class MTLMetricsAggregator:
    '''
    Aggregator for multi-task learning metrics across multiple heads.

    Calculates:
    1. Global Exact Match (GEM): Per-pixel accuracy across all active heads.
    2. Constraint Violations: Logical inconsistencies between predicted classes.
    '''

    def __init__(
        self,
        *,
        ignore_index: int,
        mtl_constraints: list[constraints.CompiledConstraint] | None = None
    ):
        '''
        Initialize the aggregator.

        Args:
            ignore_index: Index to ignore in ground truth for GEM and
                validity masks.
            constraints: Optional list of logical constraints to evaluate.
        '''

        self.ignore_index = ignore_index
        self.constraints = mtl_constraints or []

        # internal counters
        self.gem_hits: int = 0
        self.gem_samples: int = 0

        # violation counters: {constraint_name: Tally}
        self.violations: dict[str, _Tally] = {
            c.name: _Tally() for c in self.constraints
        }

    @torch.no_grad()
    def update(
        self,
        preds_1b: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor]
    ) -> None:
        '''
        Update global metrics with predictions and targets for a batch.

        Args:
            preds_1b: Predicted class IDs (1-based) per head.
            targets_1b: Ground truth labels (1-based) per head.
        '''

        # get common heads and early exit if inputs are not valid
        if not preds_1b or not targets_1b:
            return
        common_heads = [h for h in targets_1b if h in preds_1b]
        if not common_heads:
            return

        # compute global exact match
        self._get_gem(preds_1b, targets_1b, common_heads)

        # constraint violations
        self._check_violations(preds_1b, targets_1b)

    def compute(self) -> dict[str, float]:
        '''Compute final ratios for GEM and violations.'''
        results = {}
        if self.gem_samples > 0:
            results['gem'] = float(self.gem_hits) / self.gem_samples
        for name, tally in self.violations.items():
            if tally.samples > 0:
                results[f'violation_{name}'] = float(tally.hits) / tally.samples
        return results

    def reset(self) -> None:
        '''Zero all counters and move state to device if needed.'''
        self.gem_hits = 0
        self.gem_samples = 0
        for tally in self.violations.values():
            tally.hits = 0
            tally.samples = 0

    def _get_gem(
        self,
        preds_1b: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor],
        common_heads: list[str]
    ):
        '''Check GEM logic'''

        # init bool masks
        first_head = common_heads[0] # use the first head for inference
        joint_valid = torch.ones_like(targets_1b[first_head], dtype=torch.bool)
        joint_match = torch.ones_like(targets_1b[first_head], dtype=torch.bool)
        # GEM logic: Correct only if correct in ALL heads;
        # Valid only if NOT ignored in ANY head.
        for head in common_heads:
            t, p = targets_1b[head], preds_1b[head]
            joint_valid &= (t != self.ignore_index)
            joint_match &= (p == t)

        # tally
        self.gem_hits += int((joint_match & joint_valid).sum().item())
        self.gem_samples += int(joint_valid.sum().item())

    def _check_violations(
        self,
        preds_1b: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor],
    ):
        '''Check violations'''

        for c in self.constraints:
            if c.source_head not in preds_1b or c.target_head not in preds_1b:
                continue # skip if irrelavant

            p_src, p_tgt = preds_1b[c.source_head], preds_1b[c.target_head]
            t_src, t_tgt = targets_1b[c.source_head], targets_1b[c.target_head]

            # skip pixels with the ignored value
            valid_mask = (
                (t_src != self.ignore_index) &
                (t_tgt != self.ignore_index)
            )
            # create tensor of forbidden values (devide alignment)
            forbidden_tensor = torch.tensor(c.forbidden, device=p_tgt.device)
            # violations
            violation_mask = (
                (p_src == c.trigger_val) & # triggered
                torch.isin(p_tgt, forbidden_tensor) # and forbidden
            )

            # tally
            tally = self.violations[c.name]
            tally.hits += int((violation_mask & valid_mask).sum().item())
            tally.samples += int(valid_mask.sum().item())
