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
import landseg.session.engine.runtime.tasks.mtl as mtl

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
    mtl_aggregator: metrics.MTLMetricsAggregator

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

    # heads specifications
    headspecs = heads.build_headspecs(
        data_specs,
        alpha_fn=config.alpha_fn,
        en_beta=config.en_beta,
        excluded_cls=config.excluded_cls
    )

    # multihead learning constraints
    mtl_constraints = mtl.compile_constraints(config.constraints, data_specs)

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
    mtl_aggregator = metrics.MTLMetricsAggregator(
        ignore_index=data_specs.meta.label_specs.ignore_index,
        mtl_constraints=mtl_constraints
    )

    # collect components
    return EngineTasks(
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        mtl_aggregator=mtl_aggregator
    )
