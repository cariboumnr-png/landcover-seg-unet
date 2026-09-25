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

'''Unit tests for `landseg.artifacts.ledger`.'''

# local imports
import landseg.artifacts as artifacts
import landseg.artifacts.ledger as ledger


# ----- `compute_fingerprint` tests
def test_compute_fingerprint_deterministic():
    '''
    Given: Two dictionaries with identical contents in different order.
    When: Computing fingerprints.
    Then: Both produce identical SHA-256 digests.
    '''
    payload_a = {'b': 2, 'a': 1, 'nested': {'y': 20, 'x': 10}}
    payload_b = {'a': 1, 'b': 2, 'nested': {'x': 10, 'y': 20}}
    payload_c = {'a': 1, 'b': 3}

    fp_a = ledger.compute_fingerprint(payload_a)
    fp_b = ledger.compute_fingerprint(payload_b)
    fp_c = ledger.compute_fingerprint(payload_c)

    assert isinstance(fp_a, str)
    assert len(fp_a) == 64
    assert fp_a == fp_b
    assert fp_a != fp_c


# ----- `check_run_collision` tests
def test_check_run_collision_missing_manifest(tmp_path):
    '''
    Given: A non-existent runs manifest filepath.
    When: Checking for run collision.
    Then: Return None cleanly without error.
    '''
    missing_fp = str(tmp_path / 'nonexistent_manifest.json')
    result = ledger.check_run_collision('any_hash', missing_fp)
    assert result is None


def test_check_run_collision_status_filtering(tmp_path):
    '''
    Given: A runs manifest with SUCCESS, FAILED, and SKIPPED runs.
    When: Checking for collisions with different fingerprints and status.
    Then: Return colliding UID only for SUCCESS runs by default.
    '''
    manifest_fp = str(tmp_path / 'runs_manifest.json')
    manifest_data = {
        'run_success': {
            'run_uid': 'run_success',
            'fingerprint': 'hash_success',
            'status': 'SUCCESS',
        },
        'run_failed': {
            'run_uid': 'run_failed',
            'fingerprint': 'hash_failed',
            'status': 'FAILED',
        },
        'run_skipped': {
            'run_uid': 'run_skipped',
            'fingerprint': 'hash_skipped',
            'status': 'SKIPPED',
        },
    }
    artifacts.Controller[dict](manifest_fp).persist(manifest_data)

    # successful run matches
    found = ledger.check_run_collision('hash_success', manifest_fp)
    assert found == 'run_success'

    # failed run is ignored under default success_status='SUCCESS'
    found_failed = ledger.check_run_collision('hash_failed', manifest_fp)
    assert found_failed is None

    # skipped run is ignored under default success_status='SUCCESS'
    found_skipped = ledger.check_run_collision('hash_skipped', manifest_fp)
    assert found_skipped is None

    # non-existent hash returns None
    found_none = ledger.check_run_collision('unknown_hash', manifest_fp)
    assert found_none is None

    # custom success_status matches target status
    found_custom = ledger.check_run_collision(
        'hash_skipped', manifest_fp, success_status='SKIPPED'
    )
    assert found_custom == 'run_skipped'
