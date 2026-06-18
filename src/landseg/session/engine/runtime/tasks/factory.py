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
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.engine.runtime.tasks.constraints as constraints
import landseg.session.engine.runtime.tasks.heads as heads
import landseg.session.engine.runtime.tasks.loss as loss
import landseg.session.engine.runtime.tasks.metrics as metrics
import landseg.session.engine.runtime.tasks.regularization as regularization

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
    def loss_configs(self) -> loss.CompositeLossConfig: ...
    @property
    def mtl_constraints(self) -> list[constraints.MTLConstraint] | None: ...
    @property
    def mtl_reg_configs(self) -> dict[str, typing.Any]: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineTasks:
    '''Container for session-level task components.'''
    headspecs: heads.HeadSpecs
    headlosses: loss.HeadLosses
    headmetrics: metrics.HeadMetrics
    multihead_regularization: regularization.ConsistencyRegularizer
    multihead_metrics: metrics.MTLMetricsAggregator

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

    # per-head specs
    headspecs = heads.build_headspecs(
        data_specs,
        alpha_fn=config.alpha_fn,
        en_beta=config.en_beta,
        excluded_cls=config.excluded_cls
    )

    # per-head loss
    headlosses = loss.build_headlosses(
        headspecs,
        config=config.loss_configs,
        ignore_index=data_specs.meta.label_specs.ignore_index,
        spectral_band_indices=data_specs.meta.image_specs.spec_channels
    )

    # per-head segmentation metrics
    headmetrics = metrics.build_headmetrics(
        headspecs,
        ignore_index=data_specs.meta.label_specs.ignore_index
    )

    # mutli-head regularization (logical consistencies)
    # compiled constraints - 0-based indices
    mtl_regularization = regularization.ConsistencyRegularizer(
        constraints.compile_constraints(config.mtl_constraints, data_specs),
        reg_lambda=config.mtl_reg_configs.get('consistency_reg_lambda', 0.05),
        ignore_index=data_specs.meta.label_specs.ignore_index,
        reduction=config.mtl_reg_configs.get('consistency_reduction', 'mean')
    )

    # multi-head diagnostic metrics (GEM, logical violations)
    # raw constraints - 1-based indices
    mtl_metrics = metrics.MTLMetricsAggregator(
        config.mtl_constraints,
        ignore_index=data_specs.meta.label_specs.ignore_index,
    )

    # collect components
    return EngineTasks(
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        multihead_regularization=mtl_regularization,
        multihead_metrics=mtl_metrics
    )
