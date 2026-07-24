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

'''Unit tests for engine runtime builder (session/engine/runtime/builder.py).'''

# third-party imports
import pytest
# local imports
import landseg.session.engine.runtime.builder as builder_mod


# ----- `build_engine_runtime` tests
def test_build_engine_runtime_patch_size_divisibility_error(
    session_config,
    dataspecs,
    mock_model,
    mock_dataloaders
):
    '''
    Given: Dataloader patch_size (15) not divisible by model spatial_divisor (16).
    When: Calling `build_engine_runtime`.
    Then: Raise `ValueError` matching patch dimension divisibility.
    '''
    mock_dataloaders.meta.patch_size = 15
    mock_model.spatial_divisor = 16

    with pytest.raises(ValueError, match='Invalid patch dimension'):
        builder_mod.build_engine_runtime(
            dataspecs=dataspecs,
            dataloaders=mock_dataloaders,
            model=mock_model,
            config=session_config,
            device='cpu'
        )


def test_build_engine_runtime_success(
    session_config,
    dataspecs,
    mock_model,
    mock_dataloaders
):
    '''
    Given: Compatible dataloaders, model, dataspecs, and session_config.
    When: Calling `build_engine_runtime`.
    Then: Return populated `EngineRuntime` with engine, engine_optim, engine_tasks.
    '''
    runtime = builder_mod.build_engine_runtime(
        dataspecs=dataspecs,
        dataloaders=mock_dataloaders,
        model=mock_model,
        config=session_config,
        device='cpu'
    )

    assert isinstance(runtime, builder_mod.EngineRuntime)
    assert runtime.engine is not None
    assert runtime.engine_optim is not None
    assert runtime.engine_tasks is not None
