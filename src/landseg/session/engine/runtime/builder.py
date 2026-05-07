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
Engine runtime building
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.engine.runtime.executor as executor
import landseg.session.engine.runtime.optim as engine_optim
import landseg.session.engine.runtime.tasks as engine_tasks
import landseg.session.engine.protocols as protocols


# ---------------------------------Public Type---------------------------------
class EngineRuntimeConfigShape(typing.Protocol):
    '''Structural typing interface for engine building.'''
    @property
    def tasks(self) -> engine_tasks.TaskConfig: ...
    @property
    def optimization(self) -> engine_optim.OptimConfig: ...
    @property
    def runtime(self) -> common.RuntimeConfigLike: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineRuntime:
    '''Engine core components bundle.'''
    engine: executor.BatchEngine
    engine_optim: engine_optim.Optimization
    engine_tasks: engine_tasks.EngineTasks

# -------------------------------Public Function-------------------------------
def build_engine_runtime(
    *,
    dataspecs: core.DataSpecs,
    dataloaders: protocols.DataLoadersLike,
    model: core.MultiheadModelLike,
    config: EngineRuntimeConfigShape,
    device: str
) -> EngineRuntime:
    '''
    doc
    '''

    # aliases
    runtime = config.runtime
    tasks_config = config.tasks
    optim_config = config.optimization
    meta = dataloaders.meta
    preview_ctx = meta.preview_context

    # initialize engine state
    engine_state = executor.initialize_state(
        all_heads=list(dataspecs.heads.class_counts.keys()),
        batch_size=meta.batch_size,
        use_amp=runtime.precision.use_amp,
        device=device
    )

    # batch engine
    batch_config = executor.BatchEngineConfig(
        enable_logit_adjust=runtime.logit_adjust.enable,
        logit_adjust_alpha=runtime.logit_adjust.alpha,
        use_amp=runtime.precision.use_amp,
        parent_map=dataspecs.heads.head_parent,
        patch_per_blk=preview_ctx.patch_per_blk if preview_ctx else None,
        patch_per_dim=preview_ctx.patch_per_dim if preview_ctx else None,
        block_columns=preview_ctx.block_columns if preview_ctx else None
    )
    batch_engine = executor.BatchEngine(
        model,
        engine_state,
        batch_config,
        device=device
    )

    # engine optimization
    optimization = engine_optim.build_optimization(model, optim_config)

    # engine tasks
    tasks = engine_tasks.build_engine_tasks(dataspecs, tasks_config)

    # engine core bundle
    return EngineRuntime(
        engine=batch_engine,
        engine_optim=optimization,
        engine_tasks=tasks,
    )
