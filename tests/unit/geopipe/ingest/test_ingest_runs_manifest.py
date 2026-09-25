# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Unit tests for ingestion runs manifest bookkeeping and run UIDs.'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.ingest as ingest


# ----- test cases
def test_ingestion_paths_init_creates_runs_manifest(tmp_path):
    '''
    Given: IngestionPaths with an uninitialized root directory.
    When: `init_pipeline_folders` is invoked.
    Then: Create empty ingestion_runs.json with valid hash.
    '''
    ingest_paths = artifacts.IngestionPaths(str(tmp_path / 'ingest'))
    ingest_paths.init_pipeline_folders()

    assert os.path.exists(ingest_paths.runs_manifest)
    ctrl = artifacts.Controller[dict](ingest_paths.runs_manifest)
    manifest = ctrl.fetch()
    assert manifest == {}


def test_ingestion_logger_update_runs_manifest(tmp_path):
    '''
    Given: IngestionLogger with initialized summary and UID.
    When: Updating runs manifest upon run completion.
    Then: Record run record with status, folder, and lineage.
    '''
    ingest_dpath = str(tmp_path / 'ingest')
    ingest_paths = artifacts.IngestionPaths(ingest_dpath)
    ingest_paths.init_pipeline_folders()

    logger = ingest.IngestionLogger(
        name='test_ingest_manifest',
        log_file=ingest_paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id=ingest_paths.run_id)
    uid = logger.run_uid
    assert uid.startswith('ingest_')
    assert len(uid) == 23

    logger.set_harmonization_reference(
        run_uid='harmonize_1234567890abcdef',
        run_id='run_0001'
    )
    logger.set_identity('sha256_ingest_mock')

    logger.update_runs_manifest(
        ingest_paths.runs_manifest,
        ingest_paths.effective_run_folder
    )

    ctrl = artifacts.Controller[dict](ingest_paths.runs_manifest)
    manifest = ctrl.fetch()
    assert uid in manifest
    rec = manifest[uid]
    assert rec['run_uid'] == uid
    assert rec['run_id'] == 'run_0001'
    assert rec['harmonization_run_uid'] == 'harmonize_1234567890abcdef'
    assert rec['harmonization_run_id'] == 'run_0001'
    assert rec['fingerprint'] == 'sha256_ingest_mock'
    assert rec['status'] == 'SUCCESS'
    assert os.path.basename(rec['run_folder']) == 'run_0001'


def test_ingestion_runs_manifest_incremental_tracking(tmp_path):
    '''
    Given: Sequential ingestion runs across run_0001 and run_0002.
    When: Each run updates the shared runs manifest.
    Then: Both runs are persisted with unique UIDs and appropriate status.
    '''
    ingest_dpath = str(tmp_path / 'ingest')

    # run 1 - success
    paths_1 = artifacts.IngestionPaths(ingest_dpath)
    paths_1.init_pipeline_folders()
    logger_1 = ingest.IngestionLogger(
        name='run_1',
        log_file=paths_1.report,
        enable_file_log=False
    )
    logger_1.init_summary(run_id=paths_1.run_id)
    logger_1.set_harmonization_reference(
        run_uid='harmonize_run1_uid',
        run_id='run_0001'
    )
    logger_1.update_runs_manifest(
        paths_1.runs_manifest,
        paths_1.effective_run_folder
    )

    # run 2 - failed
    paths_2 = artifacts.IngestionPaths(ingest_dpath)
    paths_2.init_pipeline_folders()
    logger_2 = ingest.IngestionLogger(
        name='run_2',
        log_file=paths_2.report,
        enable_file_log=False
    )
    logger_2.init_summary(run_id=paths_2.run_id)
    logger_2.set_harmonization_reference(
        run_uid='harmonize_run2_uid',
        run_id='run_0002'
    )
    logger_2.set_summary_status('FAILED')
    logger_2.update_runs_manifest(
        paths_2.runs_manifest,
        paths_2.effective_run_folder
    )

    ctrl = artifacts.Controller[dict](paths_1.runs_manifest)
    manifest = ctrl.fetch()
    assert len(manifest) == 2

    uids = list(manifest.keys())
    assert uids[0] != uids[1]

    assert manifest[uids[0]]['run_id'] == 'run_0001'
    assert manifest[uids[0]]['harmonization_run_uid'] == 'harmonize_run1_uid'
    assert manifest[uids[0]]['status'] == 'SUCCESS'

    assert manifest[uids[1]]['run_id'] == 'run_0002'
    assert manifest[uids[1]]['harmonization_run_uid'] == 'harmonize_run2_uid'
    assert manifest[uids[1]]['status'] == 'FAILED'
