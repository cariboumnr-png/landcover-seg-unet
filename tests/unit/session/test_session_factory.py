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

'''Unit tests for session factory (session/factory.py).'''

# local imports
import landseg.session.engine.epoch as epoch_mod
import landseg.session.factory as factory_mod
import landseg.session.orchestration as orchestration_mod


# ----- `session/factory.py` builder entry point tests
def test_build_overfit_session(
    session_config,
    dataspecs,
    mock_model,
    mock_logger,
    mock_dataloaders,
    monkeypatch
):
    '''
    Given: Data specs, mock model, session config, and logger.
    When: Calling `build_overfit_session`.
    Then: Return `EpochEngine` configured in 'train_eval' mode.
    '''
    monkeypatch.setattr(
        'landseg.session.data.build_dataloaders',
        lambda *args, **kwargs: mock_dataloaders
    )
    context = factory_mod.SessionBuildContext(device='cpu')

    engine_session = factory_mod.build_overfit_session(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=context,
        logger=mock_logger  # type: ignore
    )

    assert isinstance(engine_session, epoch_mod.EpochEngine)
    assert engine_session.mode == 'train_eval'


def test_build_evaluate_session(
    session_config,
    dataspecs,
    mock_model,
    mock_logger,
    mock_dataloaders,
    monkeypatch
):
    '''
    Given: Data specs, mock model, session config, and logger.
    When: Calling `build_evaluate_session`.
    Then: Return `EpochEngine` configured in 'eval_only' mode.
    '''
    monkeypatch.setattr(
        'landseg.session.data.build_dataloaders',
        lambda *args, **kwargs: mock_dataloaders
    )
    context = factory_mod.SessionBuildContext(device='cpu')

    engine_session = factory_mod.build_evaluate_session(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=context,
        logger=mock_logger  # type: ignore
    )

    assert isinstance(engine_session, epoch_mod.EpochEngine)
    assert engine_session.mode == 'eval_only'


def test_build_continuous_training_session(
    session_config,
    dataspecs,
    mock_model,
    mock_logger,
    mock_dataloaders,
    mock_session_paths,
    monkeypatch
):
    '''
    Given: Valid session context with results paths manager.
    When: Calling `build_continous_training_session`.
    Then: Return `ContinuousRunner` orchestrator.
    '''
    monkeypatch.setattr(
        'landseg.session.data.build_dataloaders',
        lambda *args, **kwargs: mock_dataloaders
    )

    context = factory_mod.SessionBuildContext(
        device='cpu',
        session_paths=mock_session_paths
    )

    runner = factory_mod.build_continous_training_session(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=context,
        logger=mock_logger  # type: ignore
    )

    assert isinstance(runner, orchestration_mod.ContinuousRunner)
