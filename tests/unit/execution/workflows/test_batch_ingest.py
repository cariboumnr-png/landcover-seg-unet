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

'''Unit tests for batch ingestion workflow orchestration.'''

# standard imports
import pathlib
import typing
# third-party imports
import pytest
# local imports
import landseg.configs as configs
import landseg.execution.workflows.batch_ingest as batch_ingest


# ----- `execute_batch_ingest` tests
def test_execute_batch_ingest_noop_when_no_pending(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: A RootConfig with zero pending harmonization batches.
    When: Calling `execute_batch_ingest`.
    Then: Exit cleanly without instantiating `DataIngestion`.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path)
    cfg.execution.verbosity = 'silent'

    monkeypatch.setattr(
        'landseg.geopipe.resolve_pending_ingestion_batches',
        lambda *_a, **_kw: [],
    )
    called: list[typing.Any] = []

    monkeypatch.setattr(
        batch_ingest.pipelines,
        'DataIngestion',
        lambda *_a: called.append(True),
    )

    batch_ingest.execute_batch_ingest(cfg)
    assert not called


def test_execute_batch_ingest_runs_all_planned_batches(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Multiple pending harmonization records.
    When: Calling `execute_batch_ingest`.
    Then: Run `pipelines.DataIngestion(config).run(record)` for each batch.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path)
    cfg.execution.verbosity = 'silent'

    batches = [
        {'run_uid': 'u1', 'run_id': 'run_0001'},
        {'run_uid': 'u2', 'run_id': 'run_0002'},
    ]
    monkeypatch.setattr(
        'landseg.geopipe.resolve_pending_ingestion_batches',
        lambda *_a, **_kw: batches,
    )
    ran: list[typing.Any] = []

    class _FakeIngestionPipeline:
        '''Minimal DataIngestion pipeline stub capturing invoked batches.'''

        def __init__(self, pipeline_cfg: configs.RootConfig):
            assert pipeline_cfg is cfg

        def run(self, batch: typing.Any) -> None:
            '''Record processed batch record.'''
            ran.append(batch)

    monkeypatch.setattr(
        batch_ingest.pipelines,
        'DataIngestion',
        _FakeIngestionPipeline,
    )

    batch_ingest.execute_batch_ingest(cfg)
    assert ran == batches
