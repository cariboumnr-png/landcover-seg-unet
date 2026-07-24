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

'''Fixtures for testing `landseg.session.orchestration` subpackage.'''

# third-party imports
import pytest
# local imports
import landseg.core as core


# ----- mock orchestration helper classes
class DummyOptimization:
    def __init__(self):
        self.lrs = [1e-3]
        self.reconfigured = False

    def reconfigure(self, lr=None, sched_args=None):
        _ = lr, sched_args
        self.reconfigured = True


class DummyTrainer:
    def __init__(self):
        self.optimization = DummyOptimization()


class DummyEpochRunner:
    def __init__(self, step_results: core.SessionStepResults | None = None):
        self.trainer = DummyTrainer()
        self.total_train_batch = 2
        self.step_results = step_results or self._build_default_results(0.80)
        self.head_state_set = False
        self.head_state_reset = False

    def set_head_state(self, active_heads=None, frozen_heads=None):
        _ = active_heads, frozen_heads
        self.head_state_set = True

    def reset_head_state(self):
        self.head_state_reset = True

    def run_epoch(self, epoch: int) -> core.SessionStepResults:
        _ = epoch
        return self.step_results

    def _build_default_results(self, val_iou: float) -> core.SessionStepResults:
        val_res = core.ValStepResults(
            head_metrics={'head_1': core.AccumulatedMetrics(mean=val_iou)}
        )
        return core.SessionStepResults(validation=val_res)


# ----- pytest fixtures
@pytest.fixture
def dummy_epoch_runner():
    '''Return a `DummyEpochRunner` instance for orchestration policy tests.'''
    return DummyEpochRunner()
