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

'''Unit test suite for `landseg.notebooks.registry` Unity Catalog module.'''

# standard imports
import hashlib
import pathlib
# third-party imports
import mlflow
import pytest
# local imports
import landseg.notebooks as nb


def test_register_model_local_bypass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: Local non-Databricks execution environment.
    When: Calling register_unity_catalog_model.
    Then: Safely bypasses remote registration and returns metadata dictionary.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)

    result = nb.register_unity_catalog_model(
        exp_root=tmp_path,
        catalog='dev_cat',
        schema='test_schema',
        model_name='unet_backbone',
    )
    assert result['status'] == 'SKIPPED_LOCAL'
    assert result['full_model_name'] == 'dev_cat.test_schema.unet_backbone'


def test_register_model_resolves_checkpoint_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: Checkpoint and manifest files present in the experiment hierarchy.
    When: Invoking register_unity_catalog_model.
    Then: Automatically locates checkpoint and computes manifest SHA256.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)

    ckpt_dir = tmp_path / 'artifacts' / 'checkpoints'
    ckpt_dir.mkdir(parents=True)
    best_ckpt = ckpt_dir / 'phase_epoch_best.pt'
    best_ckpt.write_bytes(b'dummy_weights_data')

    manifest_file = tmp_path / 'artifacts' / 'manifest.json'
    manifest_content = b'{"dataset": "sentinel2_tiles"}'
    manifest_file.write_bytes(manifest_content)
    expected_hash = hashlib.sha256(manifest_content).hexdigest()

    result = nb.register_unity_catalog_model(exp_root=tmp_path)
    assert result['checkpoint_path'] == str(best_ckpt)
    assert result['manifest_sha256'] == expected_hash


def test_register_model_databricks_simulation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: Databricks runtime environment simulated.
    When: Registering model to Unity Catalog.
    Then: Sets databricks-uc registry URI and delegates to mlflow.register_model.
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '15.4')
    monkeypatch.setenv('DB_CLUSTER_ID', 'cluster-gpu-99')

    called_registry_uri = []
    registered_args = {}

    def fake_set_registry_uri(uri: str):
        called_registry_uri.append(uri)

    class FakeRunInfo:
        run_id = 'test_run_12345'

    class FakeRun:
        info = FakeRunInfo()

    class FakeModelVersion:
        version = '3'

    def fake_register_model(model_uri: str, name: str, tags: dict = None):
        registered_args['model_uri'] = model_uri
        registered_args['name'] = name
        registered_args['tags'] = tags
        return FakeModelVersion()

    monkeypatch.setattr(mlflow, 'set_registry_uri', fake_set_registry_uri)
    monkeypatch.setattr(mlflow, 'active_run', lambda: FakeRun())
    monkeypatch.setattr(mlflow, 'set_tags', lambda tags: None)
    monkeypatch.setattr(mlflow, 'register_model', fake_register_model)

    res = nb.register_unity_catalog_model(
        exp_root=tmp_path,
        catalog='prod_catalog',
        schema='landcover',
        model_name='sentinel_seg',
    )

    assert called_registry_uri == ['databricks-uc']
    assert registered_args['name'] == 'prod_catalog.landcover.sentinel_seg'
    assert registered_args['tags']['cluster_id'] == 'cluster-gpu-99'
    assert res.version == '3'
