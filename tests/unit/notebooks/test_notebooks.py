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
import pytest
# local imports
import landseg.notebooks as nb


# Helper fake classes for testing Databricks widgets
class _FakeWidgets:
    def __init__(self, store: dict[str, str] | None = None):
        self._store = dict(store) if store else {}

    def text(self, name: str, default: str, label: str = ''):
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
        _ = choices, label
        if name not in self._store:
            self._store[name] = str(default)

    def get(self, name: str) -> str:
        if name in self._store:
            return self._store[name]
        raise KeyError(name)

    def get_val(self, name: str) -> str | None:
        return self._store.get(name)

    def remove(self, name: str):
        self._store.pop(name, None)

    def removeAll(self):
        self._store.clear()


class _FakeDbutils:
    def __init__(self, initial: dict[str, str] | None = None):
        self.widgets = _FakeWidgets(initial)


def _create_fake_dbutils(
    initial: dict[str, str] | None = None,
) -> _FakeDbutils:
    return _FakeDbutils(initial)


# ----- `is_notebook` tests
def test_is_notebook_false_in_test_process():
    '''
    Given: Standard non-interactive test process.
    When: Probing is_notebook.
    Then: Returns False.
    '''
    assert not nb.is_notebook()


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
    bool_val = widgets.boolean('use_amp', True)
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
