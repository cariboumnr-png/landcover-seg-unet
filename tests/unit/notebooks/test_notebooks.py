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

'''Unit test suite for `landseg.notebooks` dedicated package.'''

# standard imports
import pathlib
# third-party imports
import optuna
import pytest
# local imports
import landseg.notebooks as nb


# ----- `is_databricks` tests
def test_is_databricks_false_by_default(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: No Databricks environment variables present.
    When: Probing is_databricks.
    Then: Returns False.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)
    assert not nb.is_databricks()


def test_is_databricks_true_when_runtime_set(
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: DATABRICKS_RUNTIME_VERSION environment variable set.
    When: Probing is_databricks.
    Then: Returns True.
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '15.4')
    assert nb.is_databricks()


# ----- `is_notebook` tests
def test_is_notebook_false_in_test_process():
    '''
    Given: Standard non-interactive test process.
    When: Probing is_notebook.
    Then: Returns False.
    '''
    assert not nb.is_notebook()


# ----- `resolve_exp_root` tests
def test_resolve_exp_root_explicit_path():
    '''
    Given: Explicit exp_root argument.
    When: Resolving experiment root.
    Then: Returns the explicit path as pathlib.Path.
    '''
    result = nb.resolve_exp_root('/custom/path/experiment')
    assert result == pathlib.Path('/custom/path/experiment')


def test_resolve_exp_root_env_var(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: LANDSEG_EXP_ROOT environment variable set.
    When: Resolving experiment root with default.
    Then: Returns the path configured in the environment.
    '''
    monkeypatch.setenv('LANDSEG_EXP_ROOT', '/env/storage/exp')
    result = nb.resolve_exp_root()
    assert result == pathlib.Path('/env/storage/exp')


def test_resolve_exp_root_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: No environment variable or explicit path.
    When: Resolving experiment root.
    Then: Returns the default fallback path.
    '''
    monkeypatch.delenv('LANDSEG_EXP_ROOT', raising=False)
    monkeypatch.delenv('EXP_ROOT', raising=False)
    result = nb.resolve_exp_root()
    assert result == pathlib.Path('./experiment')


def test_resolve_exp_root_dbutils_widget():
    '''
    Given: Fake dbutils with configured exp_root widget.
    When: Resolving experiment root without explicit argument.
    Then: Resolves path value from the widget.
    '''
    fake_db = _create_fake_dbutils({'exp_root': '/Volumes/cat/sch/vol/exp'})
    result = nb.resolve_exp_root(dbutils=fake_db)
    assert result == pathlib.Path('/Volumes/cat/sch/vol/exp')


def test_resolve_exp_root_create_if_missing(tmp_path: pathlib.Path):
    '''
    Given: Non-existent directory and create_if_missing=True.
    When: Resolving experiment root.
    Then: Creates the directory on disk.
    '''
    target = tmp_path / 'new_exp_dir' / 'nested'
    assert not target.exists()
    result = nb.resolve_exp_root(target, create_if_missing=True)
    assert result == target
    assert target.is_dir()


# ----- `NotebookWidgets` tests
def test_notebook_widgets_fallback_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: NotebookWidgets initialized without dbutils.
    When: Declaring text, int, float, bool, dropdown, and path widgets.
    Then: Accurately parses and returns typed default values.
    '''
    monkeypatch.delenv('LANDSEG_BATCH_SIZE', raising=False)
    widgets = nb.NotebookWidgets()
    assert not widgets.has_dbutils

    text_val = widgets.text('exp_name', 'run_01')
    int_val = widgets.int('batch_size', 16)
    float_val = widgets.float('lr', 1e-4)
    bool_val = widgets.bool('use_amp', True)
    choice_val = widgets.dropdown(
        'model',
        default='unet',
        choices=['unet', 'unetpp'],
    )
    combo_val = widgets.combobox(
        'backbone',
        default='resnet50',
        choices=['resnet34', 'resnet50'],
    )
    multi_val = widgets.multiselect(
        'bands',
        default='blue, green, red',
        choices=['blue', 'green', 'red', 'nir'],
    )
    path_val = widgets.path('out_dir', tmp_path / 'out')

    assert text_val == 'run_01'
    assert int_val == 16
    assert float_val == 0.0001
    assert bool_val is True
    assert choice_val == 'unet'
    assert combo_val == 'resnet50'
    assert multi_val == ['blue', 'green', 'red']
    assert path_val == tmp_path / 'out'

    as_dict = widgets.to_dict()
    assert as_dict['batch_size'] == 16
    assert as_dict['exp_name'] == 'run_01'


def test_notebook_widgets_env_override(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Environment variable override for a widget key.
    When: Defining widget without dbutils.
    Then: Uses the environment variable override.
    '''
    monkeypatch.setenv('LANDSEG_DEVICE', 'cuda:1')
    widgets = nb.NotebookWidgets()
    val = widgets.dropdown(
        'device',
        default='cpu',
        choices=['cpu', 'cuda', 'cuda:1'],
    )
    assert val == 'cuda:1'


def test_notebook_widgets_with_dbutils():
    '''
    Given: Active dbutils simulation.
    When: Defining and querying widgets.
    Then: Dispatches to dbutils.widgets and returns retrieved values.
    '''
    fake_db = _create_fake_dbutils({
        'batch_size': '32',
        'is_eval': 'false',
    })
    widgets = nb.NotebookWidgets(dbutils=fake_db)
    assert widgets.has_dbutils

    int_val = widgets.int('batch_size', default=16)
    bool_val = widgets.boolean('is_eval', default=True)

    assert int_val == 32
    assert bool_val is False

    widgets.remove('batch_size')
    assert fake_db.widgets.get_val('batch_size') is None


# ----- `resolve_optuna_storage` tests
def test_resolve_optuna_storage_databricks_auto_tmp(
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Running inside Databricks runtime.
    When: Resolving Optuna storage without explicit URI.
    Then: Defaults to driver-local sqlite:////tmp/optuna.db.
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '15.4')
    storage = nb.resolve_optuna_storage()
    assert storage == 'sqlite:////tmp/optuna.db'


def test_resolve_optuna_storage_local_default(
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Running outside Databricks runtime.
    When: Resolving Optuna storage without explicit URI.
    Then: Defaults to local sqlite:///optuna.db.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)
    storage = nb.resolve_optuna_storage()
    assert storage == 'sqlite:///optuna.db'


def test_resolve_optuna_storage_explicit():
    '''
    Given: Explicit postgres or rdbms storage URI.
    When: Resolving Optuna storage.
    Then: Preserves explicit URI.
    '''
    uri = 'postgresql://user:pass@host/optuna'
    assert nb.resolve_optuna_storage(uri) == uri


# ----- `resolve_notebook_io` tests
def test_resolve_notebook_io(tmp_path: pathlib.Path):
    '''
    Given: An experiment root path.
    When: Resolving unified notebook I/O layout.
    Then: Correctly structures artifacts, results, and input subpaths.
    '''
    root = tmp_path / 'my_experiment'
    layout = nb.resolve_notebook_io(
        exp_root=root,
        create_dirs=True,
    )
    assert layout.exp_root == root
    assert layout.artifacts_dir == root / 'artifacts'
    assert layout.results_dir == root / 'results'
    assert layout.input_dir == root / 'input'
    assert layout.artifacts_dir.is_dir()
    assert layout.results_dir.is_dir()


# ----- `sync_optuna_study` and `load_study_trials` tests
def test_sync_optuna_study_and_load_trials(tmp_path: pathlib.Path):
    '''
    Given: An Optuna study completed in SQLite storage.
    When: Calling sync_optuna_study and load_study_trials.
    Then: Persists summary JSON, trials CSV, and reads back dataframe.
    '''
    db_path = tmp_path / 'optuna_test.db'
    storage_uri = f'sqlite:///{db_path}'
    study = optuna.create_study(
        study_name='test_sweep',
        storage=storage_uri,
        direction='maximize',
    )
    study.optimize(lambda t: t.suggest_float('param_x', 0.0, 1.0), n_trials=3)

    res = nb.sync_optuna_study(
        study_name='test_sweep',
        exp_root=tmp_path,
        storage=storage_uri,
    )
    assert res['study_name'] == 'test_sweep'
    assert res['total_trials'] == 3
    assert pathlib.Path(res['summary_file']).is_file()
    assert pathlib.Path(res['trials_csv']).is_file()

    loaded_df = nb.load_study_trials(exp_root=tmp_path, study_name='test_sweep')
    assert len(loaded_df) == 3


# ----- private helpers
class _FakeWidgets:
    '''In-memory simulation of Databricks dbutils.widgets.'''

    def __init__(self, initial: dict[str, str] | None = None):
        '''Initialize storage with optional pre-populated widget values.'''
        self._store: dict[str, str] = dict(initial) if initial else {}

    def text(self, name: str, default: str, label: str = ''):
        '''Simulate dbutils.widgets.text.'''
        _ = label
        if name not in self._store:
            self._store[name] = str(default)

    def dropdown(
        self,
        name: str,
        default: str,
        choices: list[str],
        label: str = '',
    ):
        '''Simulate dbutils.widgets.dropdown.'''
        _ = choices, label
        if name not in self._store:
            self._store[name] = str(default)

    def combobox(
        self,
        name: str,
        default: str,
        choices: list[str],
        label: str = '',
    ):
        '''Simulate dbutils.widgets.combobox.'''
        _ = choices, label
        if name not in self._store:
            self._store[name] = str(default)

    def multiselect(
        self,
        name: str,
        default: str,
        choices: list[str],
        label: str = '',
    ):
        '''Simulate dbutils.widgets.multiselect.'''
        _ = choices, label
        if name not in self._store:
            self._store[name] = str(default)

    def get(self, name: str) -> str:
        '''Simulate dbutils.widgets.get.'''
        if name in self._store:
            return self._store[name]
        raise KeyError(name)

    def get_val(self, name: str) -> str | None:
        '''Retrieve raw value without throwing KeyError.'''
        return self._store.get(name)

    def remove(self, name: str):
        '''Simulate dbutils.widgets.remove.'''
        self._store.pop(name, None)

    def removeAll(self):  # pylint: disable=invalid-name
        '''Simulate dbutils.widgets.removeAll.'''
        self._store.clear()


class _FakeDbutils:  # pylint: disable=too-few-public-methods
    '''In-memory simulation of Databricks dbutils.'''

    def __init__(self, initial: dict[str, str] | None = None):
        '''Initialize fake dbutils with fake widgets.'''
        self.widgets = _FakeWidgets(initial)


def _create_fake_dbutils(
    initial: dict[str, str] | None = None,
) -> _FakeDbutils:
    '''Construct a fake dbutils instance with preconfigured widgets.'''
    return _FakeDbutils(initial)
