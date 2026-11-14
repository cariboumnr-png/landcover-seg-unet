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

'''Unit test suite for `landseg.mlops.env` module.'''

# third-party imports
import pytest
# local imports
import landseg.mlops as mlops


def test_is_databricks_false_by_default(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: No Databricks environment variables present.
    When: Probing is_databricks.
    Then: Returns False.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)
    assert not mlops.is_databricks()


def test_is_databricks_true_when_runtime_set(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: DATABRICKS_RUNTIME_VERSION environment variable set.
    When: Probing is_databricks.
    Then: Returns True.
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '15.4')
    assert mlops.is_databricks()


def test_is_databricks_true_when_cluster_id_set(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: DB_CLUSTER_ID environment variable set.
    When: Probing is_databricks.
    Then: Returns True.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.setenv('DB_CLUSTER_ID', 'cluster-prod-01')
    assert mlops.is_databricks()


def test_get_cluster_id_default(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: No cluster environment variable.
    When: Probing get_cluster_id.
    Then: Returns 'local-environment'.
    '''
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)
    assert mlops.get_cluster_id() == 'local-environment'


def test_get_cluster_id_configured(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: DB_CLUSTER_ID set.
    When: Probing get_cluster_id.
    Then: Returns configured cluster ID.
    '''
    monkeypatch.setenv('DB_CLUSTER_ID', 'cluster-gpu-test')
    assert mlops.get_cluster_id() == 'cluster-gpu-test'


def test_get_runtime_version_default(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: No runtime environment variable.
    When: Probing get_runtime_version.
    Then: Returns 'local'.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    assert mlops.get_runtime_version() == 'local'


def test_get_runtime_version_configured(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: DATABRICKS_RUNTIME_VERSION set.
    When: Probing get_runtime_version.
    Then: Returns configured version string.
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '14.3.x-gpu-ml')
    assert mlops.get_runtime_version() == '14.3.x-gpu-ml'


def test_get_git_commit_env(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: GIT_COMMIT environment variable set.
    When: Probing get_git_commit.
    Then: Returns the environment commit SHA.
    '''
    monkeypatch.setenv('GIT_COMMIT', 'abc123def456')
    assert mlops.get_git_commit() == 'abc123def456'


def test_get_git_commit_databricks_env(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: DATABRICKS_GIT_COMMIT environment variable set.
    When: Probing get_git_commit.
    Then: Returns the Databricks commit SHA.
    '''
    monkeypatch.delenv('GIT_COMMIT', raising=False)
    monkeypatch.setenv('DATABRICKS_GIT_COMMIT', 'db_commit_999')
    assert mlops.get_git_commit() == 'db_commit_999'


def test_get_dbutils_explicit():
    '''
    Given: Explicit dbutils object passed.
    When: Calling get_dbutils.
    Then: Returns the exact object.
    '''
    sentinel = object()
    assert mlops.get_dbutils(sentinel) is sentinel
