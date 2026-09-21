# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Epoch engine construction utilities.

Builds the epoch-level execution engine by assembling data loaders,
runtime execution components, and training/evaluation policies from
dataset metadata and configuration.

This module serves as the orchestration entry point that wires together
all components required for epoch-wise training and evaluation.
'''

# local imports
import landseg.core as core
import landseg.session.common as common
import landseg.session.engine.batch.engine as engine
import landseg.session.engine.batch.state as state


# ----- public functions
def build_batch_engine(
    dataspecs: core.DataSpecs,
    dataloaders: common.DataLoadersLike,
    model: core.MultiheadModelLike,
    config: engine.BatchExecConfigShape,
    *,
    device: str
) -> engine.BatchEngine:
    '''
    Construct the full engine runtime from model, data specifications,
    dataloaders, and configuration objects.
    '''
    # initialize engine state
    engine_state = state.initialize_state(
        all_heads=list(dataspecs.heads.class_counts.keys()),
        batch_size=dataloaders.meta.batch_size,
        use_amp=config.use_amp,
        device=device
    )

    # batch engine
    preview_ctx = dataloaders.meta.preview_context
    exec_context = engine.BatchExecContext(
        parent_map=dataspecs.heads.head_parent,
        patch_per_blk=preview_ctx.patch_per_blk if preview_ctx else None,
        patch_per_dim=preview_ctx.patch_per_dim if preview_ctx else None,
        block_columns=preview_ctx.block_columns if preview_ctx else None,
        device=device
    )
    return engine.BatchEngine(
        model=model,
        engine_state=engine_state,
        config=config,
        context=exec_context,
    )
