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

'''Unit test suite for `landseg.mlops.storage` module.'''

# standard imports
import pathlib
# third-party imports
import optuna
import pytest
# local imports
import landseg.mlops as mlops


# Helper fake classes for testing Databricks widgets
class _FakeWidgets:
    def __init__(self, store: dict[str, str] | None = None):
        self._store = dict(store) if store else {}

    def get(self, name: str) -> str:
        if name in self._store:
            return self._store[name]
        raise KeyError(name)


class _FakeDbutils:
    def __init__(self, store: dict[str, str] | None = None):
        self.widgets = _FakeWidgets(store)


# ----- `resolve_exp_root` tests
def test_resolve_exp_root_explicit_path():
    '''
    Given: Explicit exp_root argument.
    When: Resolving experiment root.
    Then: Returns the explicit path as pathlib.Path.
    '''
    result = mlops.resolve_exp_root('/custom/path/experiment')
    assert result == pathlib.Path('/custom/path/experiment')


def test_resolve_exp_root_env_var(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: LANDSEG_EXP_ROOT environment variable set.
    When: Resolving experiment root with default.
    Then: Returns the path configured in the environment.
    '''
    monkeypatch.setenv('LANDSEG_EXP_ROOT', '/env/storage/exp')
    result = mlops.resolve_exp_root()
    assert result == pathlib.Path('/env/storage/exp')


def test_resolve_exp_root_default_fallback(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: No environment variable or explicit path.
    When: Resolving experiment root.
    Then: Returns the default fallback path.
    '''
    monkeypatch.delenv('LANDSEG_EXP_ROOT', raising=False)
    monkeypatch.delenv('EXP_ROOT', raising=False)
    result = mlops.resolve_exp_root()
    assert result == pathlib.Path('./experiment')


def test_resolve_exp_root_dbutils_widget():
    '''
    Given: Fake dbutils with configured exp_root widget.
    When: Resolving experiment root without explicit argument.
    Then: Resolves path value from the widget.
    '''
    fake_db = _FakeDbutils({'exp_root': '/Volumes/cat/sch/vol/exp'})
    result = mlops.resolve_exp_root(dbutils=fake_db)
    assert result == pathlib.Path('/Volumes/cat/sch/vol/exp')


def test_resolve_exp_root_create_if_missing(tmp_path: pathlib.Path):
    '''
    Given: Non-existent directory and create_if_missing=True.
    When: Resolving experiment root.
    Then: Creates the directory on disk.
    '''
    target = tmp_path / 'new_exp_dir' / 'nested'
    assert not target.exists()
    result = mlops.resolve_exp_root(target, create_if_missing=True)
    assert result == target
    assert target.is_dir()


# ----- `resolve_optuna_storage` tests
def test_resolve_optuna_storage_explicit():
    '''
    Given: Explicit storage connection string provided.
    When: Resolving Optuna storage.
    Then: Returns explicit storage string without mutation.
    '''
    result = mlops.resolve_optuna_storage('postgresql://usr:pwd@host/db')
    assert result == 'postgresql://usr:pwd@host/db'


def test_resolve_optuna_storage_databricks_default(
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Databricks runtime detected.
    When: Resolving Optuna storage without explicit URI.
    Then: Directs SQLite to local driver scratch storage (/tmp/optuna.db).
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '15.4')
    result = mlops.resolve_optuna_storage()
    assert result == 'sqlite:////tmp/optuna.db'


def test_resolve_optuna_storage_local_default(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Standard local runtime (non-Databricks).
    When: Resolving Optuna storage without explicit URI.
    Then: Falls back to relative SQLite database.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)
    result = mlops.resolve_optuna_storage()
    assert result == 'sqlite:///optuna.db'


def test_resolve_optuna_storage_dbutils_widget():
    '''
    Given: Databricks dbutils with optuna_storage widget configured.
    When: Resolving storage string.
    Then: Retrieves URI from widget.
    '''
    fake_db = _FakeDbutils({'optuna_storage': 'sqlite:////local_disk0/study.db'})
    result = mlops.resolve_optuna_storage(dbutils=fake_db)
    assert result == 'sqlite:////local_disk0/study.db'


# ----- `sync_optuna_study` tests
def test_sync_optuna_study_persists_summary_and_trials(tmp_path: pathlib.Path):
    '''
    Given: In-memory Optuna study with several trials.
    When: Invoking sync_optuna_study.
    Then: Persists summary JSON, trials CSV, and summary metadata.
    '''
    study_name = 'test_mlops_sweep'
    storage_uri = f'sqlite:///{tmp_path / "scratch_optuna.db"}'

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction='maximize',
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        return float(1.0 - lr)

    study.optimize(objective, n_trials=3)

    summary = mlops.sync_optuna_study(
        study_name=study_name,
        exp_root=tmp_path,
        storage=storage_uri,
    )

    assert summary['study_name'] == study_name
    assert summary['total_trials'] == 3
    assert pathlib.Path(summary['summary_file']).is_file()
    assert summary['trials_csv'] is not None
    assert pathlib.Path(summary['trials_csv']).is_file()

    # Verify loading via load_study_trials
    df = mlops.load_study_trials(exp_root=tmp_path, study_name=study_name)
    assert len(df) == 3
