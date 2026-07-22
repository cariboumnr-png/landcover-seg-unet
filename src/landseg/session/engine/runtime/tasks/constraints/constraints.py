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
Compilation and validation of multi-task learning constraints.

Constraint configuration uses 1-based class IDs to match dataset labels
and other user-facing configuration. This module validates those
constraints against the configured data specifications, then compiles
them into immutable, tensor-ready constraints containing 0-based class
indices.

Each constraint describes an invalid relationship between two task
heads:

    source_head == trigger_val  AND  target_head in forbidden

Compilation validates that:

    - constraint names are unique;
    - source and target heads exist and are different;
    - trigger and forbidden class IDs are 1-based;
    - class IDs are valid for their respective heads.

The compiled constraints can be consumed directly by consistency
metrics and differentiable regularizers that index class-probability
tensors.

`compile_constraints` returns `None` when no constraints are configured.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core


# ----- input constraint config container shape
class MTLConstraint(typing.Protocol):
    @property
    def name(self) -> str: ...
    @property
    def source_head(self) -> str: ...
    @property
    def trigger_val(self) -> int: ...       # 1-based
    @property
    def target_head(self) -> str: ...
    @property
    def forbidden(self) -> list[int]: ...   # 1-based


# ----- complied constraint container
@dataclasses.dataclass(frozen=True)
class CompiledConstraint:
    '''
    Tensor-ready view of a consistency constraint.

    Constraint configuration uses 1-based class IDs to match dataset
    labels. Compiled constraints store 0-based class indices for direct
    indexing into softmax probability tensors.
    '''
    name: str
    source_head: str
    trigger_val: int                # 0-based index
    target_head: str
    forbidden: tuple[int, ...]      # 0-based indices


# ----- public API
def compile_constraints(
    mtl_constraints: typing.Sequence[MTLConstraint] | None,
    data_specs: core.DataSpecs
) -> list[CompiledConstraint] | None:
    '''Validate constraints against data specifications.'''
    # early exit if list is empty
    if mtl_constraints is None:
        return None

    # raise if constraints look duplicated (same names)
    names = [c.name for c in mtl_constraints]
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicated constraints in {names}')

    # get heads/indices as {head_name: list of 1-based indices}
    heads_idx = {
        k: list(range(1, len(v) + 1))
        for k, v, in data_specs.heads.class_counts.items()
    }

    # validate all constraints and return
    cc: list[CompiledConstraint] = []
    for c in mtl_constraints:

        _validate_constraint(c, heads_idx)
        cc.append(
            CompiledConstraint(
                name=c.name,
                source_head=c.source_head,
                trigger_val=c.trigger_val - 1,
                target_head=c.target_head,
                forbidden=tuple(v - 1 for v in c.forbidden),
            )
        )

    return cc if len(cc) > 0 else None


# ----- internal helpers
def _validate_constraint(
    contraint: MTLConstraint,
    heads_idx: dict[str, list[int]]
) -> None:
    '''Validate constrain.'''
    if contraint.source_head == contraint.target_head:
        raise ValueError(
            f'Source and target heads can not be the same:'
            f'souce: {contraint.source_head} vs target: {contraint.target_head}'
        )

    if contraint.source_head not in heads_idx:
        raise ValueError(
            f'Invalid source head: {contraint.source_head}, '
            f'allowed: {list(heads_idx.keys())}'
        )

    if contraint.trigger_val < 1:
        raise ValueError(
            f'Constraint {contraint.name} trigger_val must be 1-based; '
            f'got: {contraint.trigger_val}'
        )

    if contraint.trigger_val not in heads_idx[contraint.source_head]:
        raise ValueError(
            f'Invalid trigger value: {contraint.trigger_val}, '
            f'allowed: {heads_idx[contraint.source_head]}'
        )

    if contraint.target_head not in heads_idx:
        raise ValueError(
            f'Invalid target head: {contraint.target_head}, '
            f'allowed: {list(heads_idx.keys())}'

        )

    if not contraint.forbidden:
        raise ValueError(
            f'Constraint {contraint.name} must contain '
            f'at least one forbidden class, got None'
        )

    if any(v < 1 for v in contraint.forbidden):
        raise ValueError(
            f'Constraint {contraint.name} forbidden classes must be 1-based; '
            f'got: {contraint.forbidden}'
        )

    if not all(
        f in heads_idx[contraint.target_head] for f in contraint.forbidden
    ):
        raise ValueError(
            f'Invalid forbidden classes: {contraint.forbidden}, '
            f'allowed: {heads_idx[contraint.target_head]}'
        )
