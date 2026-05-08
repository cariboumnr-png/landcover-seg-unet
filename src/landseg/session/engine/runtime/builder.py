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
import landseg.session.engine.runtime.executor as executor
import landseg.session.engine.runtime.optim as optim
import landseg.session.engine.runtime.tasks as tasks
import landseg.session.engine.protocols as protocols


# ---------------------------------Public Type---------------------------------
class _EngineRuntimeConfigShape(typing.Protocol):
    '''Structural typing interface for engine building.'''
    @property
    def engine_exec(self) -> executor.BatchExecConfigShape: ...
    @property
    def engine_optim(self) -> optim.OptimConfig: ...
    @property
    def engine_tasks(self) -> tasks.TaskConfig: ...

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class EngineRuntime:
    '''Engine core components bundle.'''
    engine: executor.BatchEngine
    engine_optim: optim.Optimization
    engine_tasks: tasks.EngineTasks

# -------------------------------Public Function-------------------------------
def build_engine_runtime(
    *,
    dataspecs: core.DataSpecs,
    dataloaders: protocols.DataLoadersLike,
    model: core.MultiheadModelLike,
    config: _EngineRuntimeConfigShape,
    device: str
) -> EngineRuntime:
    '''
    doc
    '''

    # aliases
    exec_config = config.engine_exec
    tasks_config = config.engine_tasks
    optim_config = config.engine_optim
    meta = dataloaders.meta
    preview_ctx = meta.preview_context

    # initialize engine state
    engine_state = executor.initialize_state(
        all_heads=list(dataspecs.heads.class_counts.keys()),
        batch_size=meta.batch_size,
        use_amp=exec_config.use_amp,
        device=device
    )

    # batch engine
    exec_context = executor.BatchExecContext(
        parent_map=dataspecs.heads.head_parent,
        patch_per_blk=preview_ctx.patch_per_blk if preview_ctx else None,
        patch_per_dim=preview_ctx.patch_per_dim if preview_ctx else None,
        block_columns=preview_ctx.block_columns if preview_ctx else None
    )
    batch_engine = executor.BatchEngine(
        model,
        engine_state,
        exec_config,
        exec_context,
        device=device
    )

    # engine core bundle
    return EngineRuntime(
        engine=batch_engine,
        engine_optim=optim.build_optimization(model, optim_config),
        engine_tasks=tasks.build_engine_tasks(dataspecs, tasks_config),
    )
