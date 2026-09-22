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

'''Unit tests for session engine builder (session/engine/builder.py).'''

# third-party imports
import pytest
# local imports
import landseg.session.engine as engine_mod
import landseg.session.engine.epoch as epoch_mod


# ----- `build_engine` tests
def test_build_engine_patch_size_divisibility_error(
    session_config,
    dataspecs,
    mock_model,
    mock_dispatcher,
    mock_dataloaders,
):
    '''
    Given: Dataloader patch_size (128) indivisible by spatial_divisor (18).
    When: Calling `build_engine`.
    Then: Raise `ValueError` matching patch dimension divisibility.
    '''
    mock_model.spatial_divisor = 18
    session_config.data_loader.patch_size = 128

    context = engine_mod.EngineContext(
        dataspecs=dataspecs,
        model=mock_model,
        dispatcher=mock_dispatcher,
        device='cpu',
    )

    with pytest.raises(ValueError, match='Invalid patch dimension'):
        engine_mod.build_engine(
            mock_dataloaders,
            context,
            session_config,
            mode='train_eval',
        )


def test_build_engine_success(
    session_config,
    dataspecs,
    mock_model,
    mock_dispatcher,
    mock_dataloaders,
):
    '''
    Given: Compatible configs, model, dataspecs, and context.
    When: Calling `build_engine`.
    Then: Return populated `EpochRunner`.
    '''
    context = engine_mod.EngineContext(
        dataspecs=dataspecs,
        model=mock_model,
        dispatcher=mock_dispatcher,
        device='cpu',
    )

    runner = engine_mod.build_engine(
        mock_dataloaders,
        context,
        session_config,
        mode='train_eval',
    )

    assert isinstance(runner, epoch_mod.EpochRunner)
    assert runner.mode == 'train_eval'
    assert runner.trainer is not None
    assert runner.evaluator is not None
