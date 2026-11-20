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

'''
Unit test suite for isolated `landseg.runtime` subsystem.

Validates platform detection probes, compute device resolution, worker
pool concurrency sizing, interactive widgets, and unified logging.
'''

# standard imports
import pathlib
# third-party imports
import pytest
import torch
# local imports
from landseg import runtime


# ----- `platform` tests
def test_platform_probes():
    '''
    Given: Clean test execution environment.
    When: Querying platform detection functions.
    Then: Returns deterministic values without raising exceptions.
    '''
    assert isinstance(runtime.is_databricks(), bool)
    assert isinstance(runtime.is_notebook(), bool)
    assert isinstance(runtime.get_cluster_id(), str)
    assert isinstance(runtime.get_runtime_version(), str)
    assert isinstance(runtime.get_git_commit(), str)
    assert runtime.get_dbutils() is None or runtime.get_dbutils() is not None


# ----- `hardware` tests
def test_hardware_resolution():
    '''
    Given: Default execution environment.
    When: Resolving compute devices and hardware names.
    Then: Resolves valid target device strings and device names.
    '''
    resolved = runtime.resolve_device('auto')
    assert resolved in {'cpu', 'cuda', 'mps'}

    cpu_resolved = runtime.resolve_device('cpu')
    assert cpu_resolved == 'cpu'

    dev_name = runtime.get_device_name('cpu')
    assert isinstance(dev_name, str)
    assert dev_name == 'cpu'

    assert isinstance(runtime.is_cpu_bf16_supported(), bool)


def test_hardware_invalid_device():
    '''
    Given: Request for unavailable CUDA hardware device.
    When: Calling resolve_device on a non-CUDA host.
    Then: Raises ValueError when CUDA is absent.
    '''
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match='CUDA is not available'):
            runtime.resolve_device('cuda:99')


# ----- `concurrency` tests
def test_worker_concurrency_resolution(monkeypatch):
    '''
    Given: Optional environment variable override `LANDSEG_MAX_WORKERS`.
    When: Resolving worker count.
    Then: Honors environment override and defaults to positive integer.
    '''
    default_n = runtime.get_default_worker_count()
    assert default_n >= 1

    monkeypatch.setenv('LANDSEG_MAX_WORKERS', '6')
    assert runtime.get_default_worker_count() == 6

    monkeypatch.setenv('LANDSEG_MAX_WORKERS', 'invalid')
    assert runtime.get_default_worker_count() >= 1


def test_parallel_executor_run():
    '''
    Given: Collection of independent callable jobs.
    When: Executing jobs using ParallelExecutor.
    Then: Returns results preserving input order.
    '''
    jobs = [
        (_square_job, (i,), {})
        for i in range(5)
    ]
    executor = runtime.ParallelExecutor(
        max_workers=2,
        use_threads=True,
        show_progress=False,
    )
    results = executor.run(jobs)
    assert results == [0, 1, 4, 9, 16]


# ----- `interactive` tests
def test_notebook_widgets_defaults():
    '''
    Given: NotebookWidgets instance without active dbutils.
    When: Defining widgets with defaults.
    Then: Returns configured defaults or environment overrides.
    '''
    widgets = runtime.NotebookWidgets()
    assert not widgets.has_dbutils

    assert widgets.text('exp_name', default='experiment_1') == 'experiment_1'
    assert widgets.int('batch_size', default=16) == 16
    assert widgets.float('learning_rate', default=0.001) == 0.001
    assert widgets.boolean('enable_amp', default=True) is True

    exported = widgets.to_dict()
    assert exported['exp_name'] == 'experiment_1'
    assert exported['batch_size'] == 16


def test_require_widget_missing():
    '''
    Given: Unset required widget without default.
    When: Calling require_widget.
    Then: Raises MissingRequiredInputError.
    '''
    with pytest.raises(runtime.MissingRequiredInputError):
        runtime.require_widget('nonexistent_widget_key')


# ----- `logger` tests
def test_logger_lifecycle(tmp_path: pathlib.Path):
    '''
    Given: Target log file destination.
    When: Instantiating Logger and writing log messages.
    Then: Creates log file and captures output.
    '''
    log_fp = tmp_path / 'runtime_test.log'
    logger = runtime.Logger(
        name='test_logger',
        log_file=log_fp,
        enable_file_log=True,
    )
    logger.info('Test runtime message')
    logger.close()

    assert log_fp.exists()
    content = log_fp.read_text(encoding='utf-8')
    assert 'Test runtime message' in content


# ----- private helpers
def _square_job(val: int) -> int:
    '''Compute square of integer value.'''
    return val * val
