# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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
Cross-head consistency regularization for multi-task segmentation.

This module implements a differentiable regularizer that penalizes
probability mass assigned to invalid combinations of predictions across
multiple heads. It is intended to align training with the logical
constraint metrics introduced for multi-task learning (MTL) evaluation.

Unlike focal and dice losses, this regularizer is not a supervised
per-head classification objective. It consumes multiple head logits at
once and evaluates configured cross-head constraints, such as:

    source_head == trigger_val  AND  target_head in forbidden

The regularizer uses softmax probabilities rather than hard argmax
predictions, so invalid states can be discouraged during backpropagation
before they become discrete violation metrics.
'''

# standard imports
import dataclasses
import typing
# third-party imports
import torch
import torch.nn
import torch.nn.functional
# local imports
import landseg.session.engine.runtime.tasks.constraints as constraints

# ---------------------------------Public Type---------------------------------
class ConsistencyRegConfigShape(typing.Protocol):
    @property
    def consistency_lambda(self) -> float: ...
    @property
    def consistency_reduction(self) -> str: ...

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass(frozen=True)
class _ConstraintValue:
    '''Computed penalty and weighting terms for one active constraint.'''
    name: str
    mean: torch.Tensor              # mean invalid probability
    invalid_sum: torch.Tensor
    valid_count: torch.Tensor

# --------------------------------Public Class---------------------------------
class ConsistencyRegularizer(torch.nn.Module):
    '''
    Differentiable regularizer for invalid multi-head class combinations.

    For each configured constraint, this module computes:

        P(source_head == trigger_val) *
        P(target_head in forbidden)

    over valid pixels. The returned tensor is a reduction of those
    invalid-state probabilities across all active constraints.

    The module expects:
      - logits as a dictionary of raw model outputs, keyed by head name;
      - targets as a dictionary of 1-based ground-truth labels, keyed by
        head name.

    Targets are used only for validity masking. They do not define the
    predicted invalid state; the invalid-state probability comes entirely
    from the model's softmax outputs.
    '''

    def __init__(
        self,
        mtl_constraints: list[constraints.CompiledConstraint] | None,
        configs: ConsistencyRegConfigShape,
        *,
        ignore_index: int,
    ) -> None:
        '''
        Initialize the consistency regularizer.

        Args:
            constraints:
                Pairwise constraints using 1-based class IDs. Empty or
                None constraints make the regularizer return zero.
            ignore_index:
                Ground-truth label value to exclude from regularization.
                A pixel is valid for a constraint only when both involved
                heads are non-ignored at that pixel.
            reduction:
                Reduction mode:
                    - 'mean': mean invalid probability over all valid
                        constraint-pixel pairs.
                    - 'sum': sum invalid probabilities over all valid
                        constraint-pixel pairs.
                    - 'none': one mean invalid probability per active
                        constraint.

        Raises:
            ValueError: If reduction is unsupported, if constraint names
                are duplicated, or if a constraint contains invalid class
                IDs.
        '''

        super().__init__()

        # sanity checks - reduction methods
        if configs.consistency_reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(
                f'Invalid reduction: {configs.consistency_reduction}, '
                f'expected: ["mean", "sum", "none"]'
            )

        # sanity checks - provided constrants
        mtl_constraints = mtl_constraints or []
        names = [c.name for c in mtl_constraints]
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicated consistency constraints: {names}')

        # init attributes
        self.ignore_index = ignore_index
        self.reg_lambda = configs.consistency_lambda
        self.reduction = configs.consistency_reduction
        self.constraints = mtl_constraints

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        '''
        Compute the consistency regularization value.

        Args:
            logits:
                Mapping from head name to logits of shape [B, C, H, W].
            targets_1b:
                Mapping from head name to 1-based labels of shape
                [B, H, W]. Labels are used for ignore-index masking only.

        Returns:
            A tensor whose shape depends on the reduction:
                - scalar for 'mean' and 'sum';
                - [K] tensor for 'none', where K is the number of active
                  constraints with at least one valid pixel.

            If no constraints are active, or no valid pixels exist, a
            zero scalar is returned for 'mean'/'sum' and an empty tensor
            is returned for 'none'.
        '''

        # get reference tensor for device/dtype
        ref = next(iter(logits.values()), None)
        # exit if not valid tensors found from logits (e.g., no active heads)
        if ref is None:
            return (
                torch.empty(0) if self.reduction == 'none'
                else torch.zeros(())
            )

        # compute penalty values (constraint violations)
        values = self._constraint_values(logits, targets_1b)
        # exit if no active constraints nor valid pixels
        if not values:
            return (
                ref.new_empty((0,)) if self.reduction == 'none'
                else ref.new_zeros(())
            )

        if self.reduction == 'none':
            return torch.stack([value.mean for value in values]) * self.reg_lambda

        invalids = torch.stack([value.invalid_sum for value in values]).sum()
        if self.reduction == 'sum':
            return invalids * self.reg_lambda

        # reduction=mean
        valids = torch.stack([value.valid_count for value in values]).sum()
        return invalids / valids.clamp_min(self._eps(ref)) * self.reg_lambda

    def by_constraint(
        self,
        logits: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        '''
        Compute unreduced mean penalties keyed by constraint name.

        This helper is intended for diagnostics and logging. It mirrors
        `forward(..., reduction='none')`, but preserves constraint names
        and skips inactive constraints.

        Args:
            logits: Mapping from head name to logits [B,C,H,W].
            targets_1b: Mapping from head name to 1-based labels [B,H,W].

        Returns:
            Dictionary mapping constraint names to scalar tensors.
        '''

        return {
            value.name: value.mean
            for value in self._constraint_values(logits, targets_1b)
        }

    def _constraint_values(
        self,
        logits: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor],
    ) -> list[_ConstraintValue]:
        '''Compute valid per-constraint penalties once.'''

        values: list[_ConstraintValue] = []
        # filters out missing heads or zero-valid-pixel constraints
        for constraint in self.constraints:
            value = self._constraint_value(constraint, logits, targets_1b)
            if value is not None:
                values.append(value)
        return values

    def _constraint_value(
        self,
        constraint: constraints.CompiledConstraint,
        logits: dict[str, torch.Tensor],
        targets_1b: dict[str, torch.Tensor],
    ) -> _ConstraintValue | None:
        '''Compute the mean invalid probability for one constraint.'''

        # retrieve tensors
        source_logits = logits.get(constraint.source_head)
        target_logits = logits.get(constraint.target_head)
        source_target = targets_1b.get(constraint.source_head)
        target_target = targets_1b.get(constraint.target_head)
        # early exit if all tensors present not in batch (e.g., inactive head)
        if (
            source_logits is None
            or target_logits is None
            or source_target is None
            or target_target is None
        ):
            return None

        # tensor shape sanity checks
        self._validate_shapes(
            constraint=constraint,
            source_logits=source_logits,
            target_logits=target_logits,
            source_target=source_target,
            target_target=target_target,
        )

        # build valid pixel mask
        valid_mask = (
            (source_target != self.ignore_index) &
            (target_target != self.ignore_index)
        )
        # early exit if no valid values
        if valid_mask.sum().item() == 0:
            return None

        # compute invalid state probability & align valid mask to the result
        invalid_prob = self._invalid_state_probability(
            constraint=constraint,
            source_logits=source_logits,
            target_logits=target_logits,
        )
        valid = valid_mask.to(invalid_prob.device, invalid_prob.dtype)

        # aggregate and return
        invalid_sum = (invalid_prob * valid).sum()
        valid_count = valid.sum()
        mean = invalid_sum / valid_count.clamp_min(self._eps(invalid_prob))
        return _ConstraintValue(
            name=constraint.name,
            mean=mean,
            invalid_sum=invalid_sum,
            valid_count=valid_count,
        )

    @staticmethod
    def _validate_shapes(
        *,
        constraint: constraints.CompiledConstraint,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        source_target: torch.Tensor,
        target_target: torch.Tensor,
    ) -> None:
        '''Validate tensor ranks, spatial alignment, class indices.'''

        if source_logits.ndim != 4 or target_logits.ndim != 4:
            raise ValueError(
                f'Constraint {constraint.name} expects logits shaped '
                '[B, C, H, W].'
            )
        if source_target.ndim != 3 or target_target.ndim != 3:
            raise ValueError(
                f'Constraint {constraint.name} expects targets shaped '
                '[B, H, W].'
            )
        if source_logits.shape[0] != target_logits.shape[0]:
            raise ValueError(
                f'Constraint {constraint.name} has mismatched logit batches.'
            )
        if source_target.shape != target_target.shape:
            raise ValueError(
                f'Constraint {constraint.name} has mismatched target shapes: '
                f'{source_target.shape} vs {target_target.shape}.'
            )
        if source_logits.shape[0] != source_target.shape[0]:
            raise ValueError(
                f'Constraint {constraint.name} has mismatched source batch.'
            )
        if source_logits.shape[-2:] != source_target.shape[-2:]:
            raise ValueError(
                f'Constraint {constraint.name} has mismatched source spatial '
                'shape.'
            )
        if target_logits.shape[-2:] != target_target.shape[-2:]:
            raise ValueError(
                f'Constraint {constraint.name} has mismatched target spatial '
                'shape.'
            )
        if source_logits.shape[-2:] != target_logits.shape[-2:]:
            raise ValueError(
                f'Constraint {constraint.name} has mismatched logit spatial '
                'shape.'
            )
        if constraint.trigger_val >= source_logits.shape[1]:
            raise ValueError(
                f'Constraint {constraint.name} source class index '
                f'{constraint.trigger_val} exceeds {source_logits.shape[1]} '
                'classes.'
            )
        if max(constraint.forbidden) >= target_logits.shape[1]:
            raise ValueError(
                f'Constraint {constraint.name} forbidden class indices '
                f'{constraint.forbidden} exceed {target_logits.shape[1]} '
                'classes.'
            )

    @staticmethod
    def _invalid_state_probability(
        *,
        constraint: constraints.CompiledConstraint,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        '''Compute per-pixel probability of one invalid state.'''

        source_probs = torch.nn.functional.softmax(source_logits, dim=1)
        target_probs = torch.nn.functional.softmax(target_logits, dim=1)

        forbidden_idx = torch.as_tensor(
            constraint.forbidden,
            device=target_logits.device,
            dtype=torch.long,
        )

        source_prob = source_probs[:, constraint.trigger_val, ...]
        target_prob = target_probs.index_select(1, forbidden_idx).sum(dim=1)

        return source_prob * target_prob

    @staticmethod
    def _eps(t: torch.Tensor) -> torch.Tensor:
        '''Return dtype-safe epsilon for denominator clamping.'''

        return torch.as_tensor(
            torch.finfo(t.dtype).eps,
            dtype=t.dtype,
            device=t.device
        )
