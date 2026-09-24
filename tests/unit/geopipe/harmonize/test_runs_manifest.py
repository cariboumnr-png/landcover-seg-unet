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

'''Unit tests for harmonization runs manifest bookkeeping and run UIDs.'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize as harmonize


# ----- test cases
def test_harmonization_paths_init_creates_runs_manifest(tmp_path):
    '''
    Given: HarmonizationPaths with an uninitialized root directory.
    When: `init_pipeline_folders` is invoked.
    Then: Create empty harmonization_runs.json with valid hash.
    '''
    harm_paths = artifacts.HarmonizationPaths(str(tmp_path / 'harm'))
    harm_paths.init_pipeline_folders()

    assert os.path.exists(harm_paths.runs_manifest)
    ctrl = artifacts.Controller[dict](harm_paths.runs_manifest)
    manifest = ctrl.fetch()
    assert manifest == {}


def test_harmonization_logger_update_runs_manifest(tmp_path):
    '''
    Given: HarmonizationLogger with initialized summary and UID.
    When: Updating runs manifest upon run completion.
    Then: Record run record with status, folder, and timestamp.
    '''
    harm_dpath = str(tmp_path / 'harm')
    harm_paths = artifacts.HarmonizationPaths(harm_dpath)
    harm_paths.init_pipeline_folders()

    logger = harmonize.HarmonizationLogger(
        name='test_harm_manifest',
        log_file=harm_paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id=harm_paths.run_id)
    uid = logger.run_uid
    assert len(uid) == 16

    logger.update_runs_manifest(
        harm_paths.runs_manifest,
        harm_paths.effective_run_folder
    )

    ctrl = artifacts.Controller[dict](harm_paths.runs_manifest)
    manifest = ctrl.fetch()
    assert uid in manifest
    rec = manifest[uid]
    assert rec['run_uid'] == uid
    assert rec['run_id'] == 'run_0001'
    assert rec['status'] == 'SUCCESS'
    assert os.path.basename(rec['run_folder']) == 'run_0001'


def test_harmonization_runs_manifest_incremental_tracking(tmp_path):
    '''
    Given: Sequential harmonization runs across run_0001 and run_0002.
    When: Each run updates the shared runs manifest.
    Then: Both runs are persisted with unique UIDs and appropriate status.
    '''
    harm_dpath = str(tmp_path / 'harm')

    # run 1 - success
    paths_1 = artifacts.HarmonizationPaths(harm_dpath)
    paths_1.init_pipeline_folders()
    logger_1 = harmonize.HarmonizationLogger(
        name='run_1',
        log_file=paths_1.report,
        enable_file_log=False
    )
    logger_1.init_summary(run_id=paths_1.run_id)
    logger_1.update_runs_manifest(
        paths_1.runs_manifest,
        paths_1.effective_run_folder
    )

    # run 2 - failed
    paths_2 = artifacts.HarmonizationPaths(harm_dpath)
    paths_2.init_pipeline_folders()
    logger_2 = harmonize.HarmonizationLogger(
        name='run_2',
        log_file=paths_2.report,
        enable_file_log=False
    )
    logger_2.init_summary(run_id=paths_2.run_id)
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
    assert manifest[uids[0]]['status'] == 'SUCCESS'

    assert manifest[uids[1]]['run_id'] == 'run_0002'
    assert manifest[uids[1]]['status'] == 'FAILED'
