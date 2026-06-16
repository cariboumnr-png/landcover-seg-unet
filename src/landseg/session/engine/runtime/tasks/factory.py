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
Engine task construction utilities.

Builds session-scoped task components, including head specifications,
loss modules, and metric accumulators, from dataset metadata and user
configuration.

This module serves as the entry point for assembling all per-head
training and evaluation artifacts used by the execution engine.
'''

# standard imports
from __future__ import annotations
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.engine.runtime.tasks.heads as heads
import landseg.session.engine.runtime.tasks.loss as loss
import landseg.session.engine.runtime.tasks.metrics as metrics
import landseg.session.engine.runtime.tasks.mtl.aggregator as mtl

# ---------------------------------Public Type---------------------------------
class TaskConfigShape(typing.Protocol):
    '''Configuration interface for constructing per-head tasks.'''
    @property
    def alpha_fn(self) -> str: ...
    @property
    def en_beta(self) -> float: ...
    @property
    def excluded_cls(self) -> dict[str, list[int]] | None: ...
    @property
    def loss_types(self) -> loss.CompositeLossConfig: ...
    @property
    def constraints(self) -> list[mtl.MTLConstraint] | None: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineTasks:
    '''Container for session-level task components.'''
    headspecs: heads.HeadSpecs
    headlosses: loss.HeadLosses
    headmetrics: metrics.HeadMetrics
    mtl_aggregator: mtl.MTLMetricsAggregator

# -------------------------------Public Function-------------------------------
def build_engine_tasks(
    data_specs: core.DataSpecs,
    config: TaskConfigShape,
 ) -> EngineTasks:
    '''
    Construct per-head specifications, losses, and metrics.

    Builds all task-related components required for execution from
    dataset metadata and configuration, including head definitions, loss
    composition modules, and metric accumulators.

    Args:
        data_specs: Dataset specification containing class statistics,
            hierarchy, and metadata.
        config: Task configuration controlling loss weighting, alpha
            computation, and enabled loss types.

    Returns:
        EngineTasks:
            Container with initialized head specifications, loss modules,
            and metrics.

    Notes:
        - Head specifications are derived from dataset metadata.
        - Loss modules are composed based on configured loss types.
        - Metric modules are initialized per head with ignore-index\
          handling and optional exclusions.
    '''

    # task - heads specifications
    headspecs = heads.build_headspecs(
        data_specs,
        alpha_fn=config.alpha_fn,
        en_beta=config.en_beta,
        excluded_cls=config.excluded_cls
    )
    # task - heads loss modules
    headlosses = loss.build_headlosses(
        headspecs,
        config=config.loss_types,
        ignore_index=data_specs.meta.label_specs.ignore_index,
        spectral_band_indices=data_specs.meta.image_specs.spec_channels
    )
    # task - heads metric modules
    headmetrics = metrics.build_headmetrics(
        headspecs,
        ignore_index=data_specs.meta.label_specs.ignore_index
    )

    # task - mtl aggregator (GEM and logical constraints)
    mtl_aggregator = mtl.MTLMetricsAggregator(
        ignore_index=data_specs.meta.label_specs.ignore_index,
        constraints=_validate_constraints(config.constraints, data_specs)
    )

    # collect components
    return EngineTasks(
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        mtl_aggregator=mtl_aggregator
    )

# ------------------------------private  function------------------------------
def _validate_constraints(
    constraints: list[mtl.MTLConstraint] | None,
    data_specs: core.DataSpecs
) -> list[mtl.MTLConstraint] | None:
    '''Validate constraints against data specifications.'''

    # early exit if list is empty
    if constraints is None:
        return None

    # raise if constraints look duplicated (same names)
    names = [c.name for c in constraints]
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicated constraints in {names}')

    # get heads/indices as {head_name: list of 1-based indices}
    heads_idx = {
        k: list(range(1, len(v) + 1))
        for k, v, in data_specs.heads.class_counts.items()
    }

    # validate all constraints and return
    for c in constraints:

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
        if not all(f in heads_idx[c.target_head] for f in c.forbidden):
            raise ValueError(
                f'Invalid forbidden classes: {c.forbidden}, '
                f'allowed: {heads_idx[c.target_head]}'
            )

    return constraints
