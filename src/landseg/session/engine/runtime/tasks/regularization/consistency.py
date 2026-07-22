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
        return invalids / valids.clamp_min(_eps(ref)) * self.reg_lambda

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
            value = _constraint_value(
                constraint,
                logits,
                targets_1b,
                ignore_index=self.ignore_index
            )
            if value is not None:
                values.append(value)
        return values


# ----- internal helpers
def _constraint_value(
    constraint: constraints.CompiledConstraint,
    logits: dict[str, torch.Tensor],
    targets_1b: dict[str, torch.Tensor],
    *,
    ignore_index: int
) -> _ConstraintValue | None:
    '''Compute the mean invalid probability for one constraint.'''

    # retrieve tensors
    source_logits = logits.get(constraint.source_head)
    target_logits = logits.get(constraint.target_head)
    source_labels = targets_1b.get(constraint.source_head)
    target_labels = targets_1b.get(constraint.target_head)
    # early exit if not all tensors present in batch (e.g., inactive head)
    if (
        source_logits is None
        or target_logits is None
        or source_labels is None
        or target_labels is None
    ):
        return None

    # tensor shape sanity checks
    _validate_shapes(
        constraint=constraint,
        source_logits=source_logits,
        target_logits=target_logits,
        source_labels=source_labels,
        target_labels=target_labels,
    )

    # build valid pixel mask
    valid_mask = (
        (source_labels != ignore_index) &
        (target_labels != ignore_index)
    )
    # early exit if no valid values
    if valid_mask.sum().item() == 0:
        return None

    # compute invalid state probability & align valid mask to the result
    invalid_prob = _invalid_state_probability(
        trigger_val=constraint.trigger_val,
        forbidden=constraint.forbidden,
        source_logits=source_logits,
        target_logits=target_logits,
    )
    valid = valid_mask.to(invalid_prob.device, invalid_prob.dtype)

    # aggregate and return
    invalid_sum = (invalid_prob * valid).sum()
    valid_count = valid.sum()
    mean = invalid_sum / valid_count.clamp_min(_eps(invalid_prob))
    return _ConstraintValue(
        name=constraint.name,
        mean=mean,
        invalid_sum=invalid_sum,
        valid_count=valid_count,
    )


def _validate_shapes(
    *,
    constraint: constraints.CompiledConstraint,
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> None:
    '''Validate tensor ranks, spatial alignment, class indices.'''
    cons = f'Constraint {constraint.name}'

    # ensure both source and target tensors of the corrent shape/size/spatial
    _validate_head_pair(f'{cons} source', source_logits, source_labels)
    _validate_head_pair(f'{cons} target', target_logits, target_labels)

    # additional checks between source and target
    if source_logits.shape[0] != target_logits.shape[0]:
        raise ValueError(f'{cons}: source and target have mismatched batches.')

    if source_logits.shape[-2:] != target_logits.shape[-2:]:
        raise ValueError(f'{cons}: source and target have mismatched H * W.')

    # constraint vs logits channel
    if constraint.trigger_val >= source_logits.shape[1]:
        raise IndexError(
            f'{cons} source trigger class index {constraint.trigger_val} '
            f'exceeds {source_logits.shape[1]} classes.'
        )

    if max(constraint.forbidden) >= target_logits.shape[1]:
        raise IndexError(
            f'{cons} forbidden class indices {constraint.forbidden}'
            f' exceed {target_logits.shape[1]} classes.'
        )


def _validate_head_pair(
    name: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
):
    '''Validate dim and shape of a logits-labels tensor pair.'''
    if logits.ndim != 4:
        raise ValueError(f'{name} logits must be [B,C,H,W]')

    if labels.ndim != 3:
        raise ValueError(f'{name} labels must be [B,H,W]')

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f'{name} batch size mismatch between logits & labels')

    if logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError(f'{name} H * W mismatch between logits & labels')


def _invalid_state_probability(
    *,
    trigger_val: int,
    forbidden: tuple[int, ...],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    '''Compute per-pixel probability of one invalid state.'''
    # softmax probability from logits [B, C, H, W]
    source_probs = torch.nn.functional.softmax(source_logits, dim=1)
    target_probs = torch.nn.functional.softmax(target_logits, dim=1)
    # tuple[int, ...] -> tensor
    device = target_logits.device # match
    forbidden_idx = torch.as_tensor(forbidden, dtype=torch.long, device=device)
    # at dim C
    source_prob = source_probs[:, trigger_val, ...]
    target_prob = target_probs.index_select(1, forbidden_idx).sum(dim=1)
    return source_prob * target_prob


def _eps(t: torch.Tensor) -> torch.Tensor:
    '''Return dtype-safe epsilon for denominator clamping.'''
    eps = torch.finfo(t.dtype).eps
    return torch.as_tensor(eps, dtype=t.dtype, device=t.device)
