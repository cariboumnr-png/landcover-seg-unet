# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Unit tests for diagnose-overfit workflow orchestration.'''

# standard imports
import json
import pathlib
import types
import typing
# third-party imports
import pytest
# local imports
import landseg._constants as c
import landseg.configs as configs
import landseg.execution.workflows.diagnose_overfit as diagnose_overfit


# ----- helpers
class _DummyEpochResults:  # pylint: disable=too-few-public-methods
    '''Minimal stub matching `EpochResults` contract used by overfit loop.'''

    def __init__(self, loss: float, iou: float):
        self.training = types.SimpleNamespace(total_objective=loss)
        self.validation = types.SimpleNamespace()
        self.target_metrics = iou

    def track(self, _metric_name: str, _track_heads: object) -> None:
        '''Record tracked metrics (no-op for stub).'''
        return None


class _FakeRunner:
    '''Minimal session runner stub returning scripted per-epoch outcomes.'''

    def __init__(self, epochs: list[_DummyEpochResults]):
        self._epochs = epochs
        self.calls: dict[str, typing.Any] = {'epochs': 0, 'active_heads': None}

    def set_head_state(self, *, active_heads: list[str]) -> None:
        '''Capture active heads configured by the workflow.'''
        self.calls['active_heads'] = active_heads

    def run_epoch(self, ep: int) -> _DummyEpochResults:
        '''Return scripted epoch metrics and advance call count.'''
        self.calls['epochs'] += 1
        idx = min(ep - 1, len(self._epochs) - 1)
        return self._epochs[idx]


def _build_config(exp_root: pathlib.Path) -> configs.RootConfig:
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(exp_root)
    cfg.execution.verbosity = 'silent'
    return cfg


def _patch_overfit_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runner: _FakeRunner | None = None,
) -> None:
    dummy_specs = types.SimpleNamespace(
        heads=types.SimpleNamespace(class_counts={'landcover': [1, 1]})
    )
    monkeypatch.setattr(
        'landseg.geopipe.build_overfit_dataspec',
        lambda **_kw: dummy_specs,
    )
    monkeypatch.setattr(
        'landseg.models.build_multihead_unet',
        lambda **_kw: object(),
    )
    monkeypatch.setattr(
        'landseg.runtime.resolve_device',
        lambda _dev: 'cpu',
    )
    if runner is not None:
        monkeypatch.setattr(
            'landseg.session.build_session_runner',
            lambda **_kw: runner,
        )


# ----- `execute_diagnose_overfit` tests
def test_execute_diagnose_overfit_stops_on_target_iou(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: A valid `RootConfig` and mock session runner reaching IoU >= 0.99
        on epoch 2.
    When: Running `execute_diagnose_overfit`.
    Then: Stop early at epoch 2 and record `SUCCESS` status in summary JSON.
    '''
    cfg = _build_config(tmp_path)
    runner = _FakeRunner(
        [_DummyEpochResults(0.4, 0.72), _DummyEpochResults(0.01, 0.995)]
    )
    _patch_overfit_runtime(monkeypatch, runner)

    diagnose_overfit.execute_diagnose_overfit(cfg)

    assert runner.calls == {'epochs': 2, 'active_heads': ['landcover']}
    summary_fp = (
        tmp_path / 'results' / 'overfit_test' / 'log' / 'overfit_summary.json'
    )
    summary = json.loads(summary_fp.read_text(encoding='utf-8'))
    assert summary['status'] == 'SUCCESS'
    assert summary['results'] == {
        'final_epoch': 2,
        'final_loss': 0.01,
        'final_iou': 0.995,
        'overfit_reached': True,
    }


def test_execute_diagnose_overfit_max_epochs_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: A session failing to hit IoU >= 0.99 within max epochs, and a
        separate dataspec failure.
    When: Running `execute_diagnose_overfit`.
    Then: Log `FAILED` on threshold miss and propagate builder exceptions.
    '''
    cfg = _build_config(tmp_path)
    monkeypatch.setattr(c, 'OVERFIT_MAX_EPOCH', 3)
    runner = _FakeRunner([_DummyEpochResults(0.3, 0.55)])
    _patch_overfit_runtime(monkeypatch, runner)

    diagnose_overfit.execute_diagnose_overfit(cfg)
    summary_fp = (
        tmp_path / 'results' / 'overfit_test' / 'log' / 'overfit_summary.json'
    )
    summary = json.loads(summary_fp.read_text(encoding='utf-8'))
    assert summary['status'] == 'FAILED'
    assert summary['results']['overfit_reached'] is False
    assert summary['results']['final_epoch'] == 3

    def _boom(**_kw):
        raise ValueError('No valid block for testing is found')

    monkeypatch.setattr('landseg.geopipe.build_overfit_dataspec', _boom)
    with pytest.raises(ValueError, match='No valid block'):
        diagnose_overfit.execute_diagnose_overfit(cfg)
