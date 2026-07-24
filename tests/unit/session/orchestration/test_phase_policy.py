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

# pylint: disable=protected-access

'''Unit tests for phase policy module (policy/phase.py).'''

# local imports
import landseg.core as core
import landseg.session.orchestration.events as events
import landseg.session.orchestration.policy.phase as phase_mod


# ----- `TrackingConfig` tests
def test_tracking_config_defaults():
    '''
    Given: Default instantiation of `TrackingConfig`.
    When: Reading initial attributes.
    Then: Standard metric defaults and patience settings are present.
    '''
    cfg = phase_mod.TrackingConfig()

    assert cfg.metric_name == 'iou'
    assert cfg.track_mode == 'max'
    assert cfg.enable_early_stop is False
    assert cfg.patience_epochs == 5


# ----- `PhasePolicy` execution tests
def test_phase_policy_execute_direct(dummy_epoch_runner, session_config):
    '''
    Given: `PhasePolicy` executed via direct `execute()`.
    When: Running all phase epochs.
    Then: Return list of `SessionStepResults` for each epoch.
    '''
    phase_cfg = session_config.orchestration.single_phase
    phase_cfg.num_epochs = 2
    phase_cfg.active_heads = ['head_1']

    phase = phase_mod.PhasePolicy(
        epoch_runner=dummy_epoch_runner,
        phase_config=phase_cfg,
        track_config=phase_mod.TrackingConfig()
    )

    results = phase.execute()

    assert len(results) == 2
    assert isinstance(results[0], core.SessionStepResults)


def test_phase_policy_run_generator_flow(dummy_epoch_runner, session_config):
    '''
    Given: `PhasePolicy` executed via generator `run()`.
    When: Stepping through generator events.
    Then: Yield `PhaseStart`, `CheckpointRequest`, `MetricsReport`, `PhaseEnd`.
    '''
    phase_cfg = session_config.orchestration.single_phase
    phase_cfg.num_epochs = 1
    phase_cfg.active_heads = ['head_1']

    phase = phase_mod.PhasePolicy(
        epoch_runner=dummy_epoch_runner,
        phase_config=phase_cfg,
        track_config=phase_mod.TrackingConfig()
    )

    events_emitted = []
    best_value = 0.0
    gen = phase.run()
    try:
        while True:
            evt = next(gen)
            events_emitted.append(evt)
    except StopIteration as stop:
        best_value = stop.value

    assert any(isinstance(e, events.PhaseStart) for e in events_emitted)
    assert any(isinstance(e, events.CheckpointRequest) for e in events_emitted)
    assert any(isinstance(e, events.MetricsReport) for e in events_emitted)
    assert any(isinstance(e, events.PhaseEnd) for e in events_emitted)
    assert best_value == 0.80


def test_phase_policy_early_stopping(dummy_epoch_runner, session_config):
    '''
    Given: `PhasePolicy` with `enable_early_stop=True` and `patience_epochs=2`.
    When: Metric fails to improve over consecutive epochs.
    Then: Emit `StopRun` event and stop execution early.
    '''
    phase_cfg = session_config.orchestration.single_phase
    phase_cfg.num_epochs = 5
    phase_cfg.active_heads = ['head_1']

    track_cfg = phase_mod.TrackingConfig(
        enable_early_stop=True,
        patience_epochs=2,
        delta=0.01
    )

    phase = phase_mod.PhasePolicy(
        epoch_runner=dummy_epoch_runner,
        phase_config=phase_cfg,
        track_config=track_cfg
    )

    events_emitted = []
    gen = phase.run()
    try:
        while True:
            evt = next(gen)
            events_emitted.append(evt)
    except StopIteration:
        pass

    assert any(isinstance(e, events.StopRun) for e in events_emitted)
    assert phase.patience_reached is True
