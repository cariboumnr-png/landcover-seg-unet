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

'''Unit tests for callback dispatcher and builder (dispatcher.py, builder.py).'''

# third-party imports
import pytest
# local imports
import landseg.core as core
import landseg.session.instrumentation.callbacks as cb_mod
import landseg.session.instrumentation.callbacks.builder as builder_mod
import landseg.session.instrumentation.callbacks.dispatcher as disp_mod


# ----- mock callback helper
class MockCallback(cb_mod.BaseCallback):
    '''Mock callback tracking event invocations.'''
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def on_batch_begin(self, action: str, bidx: int) -> None:
        _ = action, bidx
        self.calls.append('on_batch_begin')

    def on_train_policy_end(self, results: core.TrainStepResults) -> None:
        _ = results
        self.calls.append('on_train_policy_end')

    def on_val_policy_end(self, results: core.ValStepResults) -> None:
        _ = results
        self.calls.append('on_val_policy_end')

    def on_session_end(self) -> None:
        self.calls.append('on_session_end')


# ----- `CallbackDispatcher` tests
def test_dispatcher_register_and_deregister():
    '''
    Given: Empty `CallbackDispatcher` and a `MockCallback`.
    When: Registering and deregistering callback.
    Then: Callback list reflects registration status.
    '''
    dispatcher = disp_mod.CallbackDispatcher()
    mock_cb = MockCallback()

    dispatcher.register(mock_cb)
    assert mock_cb in dispatcher.callbacks

    # duplicate register is ignored
    dispatcher.register(mock_cb)
    assert len(dispatcher.callbacks) == 1

    dispatcher.deregister(mock_cb)
    assert mock_cb not in dispatcher.callbacks


def test_dispatcher_event_broadcasting():
    '''
    Given: `CallbackDispatcher` with registered `MockCallback`.
    When: Broadcasting lifecycle events (`on_batch_begin`, `on_session_end`, etc.).
    Then: All registered callbacks receive event notifications.
    '''
    mock_cb = MockCallback()
    dispatcher = disp_mod.CallbackDispatcher([mock_cb])

    dispatcher.on_batch_begin('Training', 1)
    dispatcher.on_train_policy_end(core.TrainStepResults())
    dispatcher.on_val_policy_end(core.ValStepResults())
    dispatcher.on_session_end()

    assert mock_cb.calls == [
        'on_batch_begin',
        'on_train_policy_end',
        'on_val_policy_end',
        'on_session_end'
    ]


# ----- `build_dispatcher` factory tests
def test_build_dispatcher_default():
    '''
    Given: No trackers specified for `build_dispatcher`.
    When: Building dispatcher.
    Then: Dispatcher is created with standard logging and tracking callbacks.
    '''
    dispatcher = builder_mod.build_dispatcher(verbose=False)

    assert isinstance(dispatcher, disp_mod.CallbackDispatcher)
    assert len(dispatcher.callbacks) == 4


def test_build_dispatcher_tensorboard_missing_uri():
    '''
    Given: 'tb' in trackers without URI.
    When: Building dispatcher.
    Then: Raise `AssertionError` for missing URI.
    '''
    with pytest.raises(AssertionError, match='URI not provided'):
        builder_mod.build_dispatcher(trackers=['tb'], uri=None)
