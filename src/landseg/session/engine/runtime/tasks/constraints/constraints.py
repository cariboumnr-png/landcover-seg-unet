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
doc
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core

# --------------------------------Public Type----------------------------------
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

# ------------------------------Public  Dataclass------------------------------
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

# -------------------------------Public Function-------------------------------
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

        if c.source_head == c.target_head:
            raise ValueError(
                f'Source and target heads can not be the same:'
                f'souce: {c.source_head} vs target: {c.target_head}'
            )

        if c.source_head not in heads_idx:
            raise ValueError(
                f'Invalid source head: {c.source_head}, '
                f'allowed: {list(heads_idx.keys())}'
            )

        if c.trigger_val < 1:
            raise ValueError(
                f'Constraint {c.name} trigger_val must be 1-based; '
                f'got: {c.trigger_val}'
            )

        if c.trigger_val not in heads_idx[c.source_head]:
            raise ValueError(
                f'Invalid trigger value: {c.trigger_val}, '
                f'allowed: {heads_idx[c.source_head]}'
            )

        if c.target_head not in heads_idx:
            raise ValueError(
                f'Invalid target head: {c.target_head}, '
                f'allowed: {list(heads_idx.keys())}'

            )

        if any(v < 1 for v in c.forbidden):
            raise ValueError(
                f'Constraint {c.name} forbidden classes must be 1-based; '
                f'got: {c.forbidden}'
        )

        if not all(f in heads_idx[c.target_head] for f in c.forbidden):
            raise ValueError(
                f'Invalid forbidden classes: {c.forbidden}, '
                f'allowed: {heads_idx[c.target_head]}'
            )

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
