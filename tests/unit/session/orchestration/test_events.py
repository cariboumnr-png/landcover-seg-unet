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

'''Unit tests for orchestration events module (events.py).'''

# third-party imports
import pytest
# local imports
import landseg.core as core
import landseg.session.orchestration.events as events_mod


# ----- `Event` base class tests
def test_base_event_instantiation():
    '''
    Given: Name and payload for `Event`.
    When: Instantiating `Event`.
    Then: Event is immutable and holds name and payload dict.
    '''
    event = events_mod.Event(name='custom_event', payload={'key': 'val'})

    assert event.name == 'custom_event'
    assert event.payload == {'key': 'val'}

    with pytest.raises(AttributeError):
        event.name = 'new_name'  # type: ignore


# ----- phase-level event tests
def test_phase_start_and_end_events():
    '''
    Given: Phase name for `PhaseStart` and `PhaseEnd`.
    When: Instantiating phase events.
    Then: Payloads and attributes contain phase name.
    '''
    p_start = events_mod.PhaseStart(phase_name='phase_1')
    p_end = events_mod.PhaseEnd(phase_name='phase_1')

    assert p_start.name == 'phase_start'
    assert p_start.phase_name == 'phase_1'
    assert p_start.payload == {'phase_name': 'phase_1'}

    assert p_end.name == 'phase_end'
    assert p_end.phase_name == 'phase_1'
    assert p_end.payload == {'phase_name': 'phase_1'}


# ----- epoch-level event tests
def test_epoch_start_and_end_events():
    '''
    Given: Epoch index and phase name for `EpochStart` and `EpochEnd`.
    When: Instantiating epoch events.
    Then: Payloads and attributes store epoch index and phase name.
    '''
    e_start = events_mod.EpochStart(epoch_index=2, phase_name='phase_1')
    e_end = events_mod.EpochEnd(epoch_index=2, phase_name='phase_1')

    assert e_start.name == 'epoch_start'
    assert e_start.epoch_index == 2
    assert e_start.phase_name == 'phase_1'

    assert e_end.name == 'epoch_end'
    assert e_end.epoch_index == 2
    assert e_end.phase_name == 'phase_1'


# ----- control and report event tests
def test_control_and_report_events():
    '''
    Given: Parameters for `MetricsReport`, `StopRun`, and `CheckpointRequest`.
    When: Instantiating control and report events.
    Then: Custom payloads and attributes are populated.
    '''
    raw_results = core.SessionStepResults()
    m_report = events_mod.MetricsReport(
        best_so_far=0.85,
        best_epoch=3,
        is_best_epoch=True,
        raw_metrics=raw_results
    )
    stop_event = events_mod.StopRun(reason='Patience limit reached')
    ckpt_event = events_mod.CheckpointRequest(tag='best')

    assert m_report.name == 'tracking_report'
    assert m_report.best_so_far == 0.85
    assert m_report.is_best_epoch is True
    assert m_report.raw_metrics is raw_results

    assert stop_event.name == 'stop_run'
    assert stop_event.reason == 'Patience limit reached'

    assert ckpt_event.name == 'checkpoint_request'
    assert ckpt_event.tag == 'best'
