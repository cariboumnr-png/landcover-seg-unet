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
Engine core building
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.engine.batch.executor as executor
import landseg.session.engine.batch.optim as optim
import landseg.session.engine.batch.tasks as tasks
import landseg.session.engine.protocols as protocols


# ---------------------------------Public Type---------------------------------
class BatchEngineConfigShape(typing.Protocol):
    '''Structural typing interface for engine building.'''
    @property
    def tasks(self) -> tasks.TaskConfig: ...
    @property
    def optimization(self) -> optim.OptimConfig: ...
    @property
    def runtime(self) -> common.RuntimeConfigLike: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class BatchEngine:
    '''Engine core components bundle.'''
    engine: executor.BatchExecutionEngine
    engine_optim:optim.Optimization
    engine_tasks: tasks.EngineTasks

# -------------------------------Public Function-------------------------------
def build_batch_engine(
    *,
    dataspecs: core.DataSpecs,
    dataloaders: protocols.DataLoadersLike,
    model: core.MultiheadModelLike,
    config: BatchEngineConfigShape,
    device: str
) -> BatchEngine:
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
    batch_config = executor.BatchExecutorConfig(
        parent_map=dataspecs.heads.head_parent,
        use_amp=runtime.precision.use_amp,
        patch_per_blk=preview_ctx.patch_per_blk if preview_ctx else None,
        patch_per_dim=preview_ctx.patch_per_dim if preview_ctx else None,
        block_columns=preview_ctx.block_columns if preview_ctx else None
    )
    batch_executor = executor.BatchExecutionEngine(
        model,
        engine_state,
        batch_config,
        device=device
    )

    # config model logit adjustment
    batch_executor.model.set_logit_adjust_enabled(runtime.logit_adjust.enable)
    batch_executor.model.set_logit_adjust_alpha(runtime.logit_adjust.alpha)

    # engine tasks
    engine_tasks = tasks.build_engine_tasks(dataspecs, tasks_config)

    # engine optimization
    optimization = optim.build_optimization(model, optim_config)

    # engine core bundle
    return BatchEngine(
        engine=batch_executor,
        engine_optim=optimization,
        engine_tasks=engine_tasks,
    )
