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

'''Unit test suite for `landseg.mlops.registry` Unity Catalog module.'''

# standard imports
import hashlib
import pathlib
# third-party imports
import mlflow
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.mlops as mlops
import landseg.session as session


def test_register_model_local_bypass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: Local non-Databricks execution environment.
    When: Calling register_unity_catalog_model.
    Then: Bypass remote registration and return metadata dictionary.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.delenv('DB_CLUSTER_ID', raising=False)

    result = mlops.register_unity_catalog_model(
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
    Given: Multi-phase session run with `CheckpointCatalog` and manifest.
    When: Invoking register_unity_catalog_model.
    Then: Resolve `last_phase_best` final checkpoint and manifest SHA256.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)

    run = artifacts.SessionPaths.allocate_run(
        tmp_path / 'results'
    ).init_pipeline_folders()
    p1_best = run.best_checkpoint('freeze_epoch')
    p1_last = run.last_checkpoint('freeze_epoch')
    p2_best = run.best_checkpoint('finetune_epoch')
    p2_last = run.last_checkpoint('finetune_epoch')
    for fp in (p1_best, p1_last, p2_best, p2_last):
        fp.write_bytes(b'dummy_weights_data')

    catalog = session.CheckpointCatalog(run=run)
    catalog.record_save(
        'freeze', last_fpath=p1_last, epoch=1, best_fpath=p1_best, metric=0.95
    )
    catalog.mark_phase_completed('freeze')
    catalog.record_save(
        'finetune', last_fpath=p2_last, epoch=1, best_fpath=p2_best, metric=0.70
    )
    catalog.mark_phase_completed('finetune')
    catalog.persist()

    artifacts_dir = tmp_path / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = artifacts_dir / 'manifest.json'
    manifest_content = b'{"dataset": "sentinel2_tiles"}'
    manifest_file.write_bytes(manifest_content)
    expected_hash = hashlib.sha256(manifest_content).hexdigest()

    result = mlops.register_unity_catalog_model(exp_root=tmp_path)
    assert result['checkpoint_path'] == str(p2_best)
    assert result['manifest_sha256'] == expected_hash


def test_register_model_databricks_simulation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: Databricks runtime environment simulated.
    When: Registering model to Unity Catalog.
    Then: Set `databricks-uc` registry URI and delegate to
        `mlflow.register_model`.
    '''
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '15.4')
    monkeypatch.setenv('DB_CLUSTER_ID', 'cluster-gpu-99')

    called_registry_uri = []
    registered_args = {}

    def fake_set_registry_uri(uri: str):
        called_registry_uri.append(uri)

    class FakeRunInfo:
        '''Fake MLflow run info stub.'''
        run_id = 'test_run_12345'

    class FakeRun:
        '''Fake MLflow run stub.'''
        info = FakeRunInfo()

    class FakeModelVersion:
        '''Fake MLflow model version stub.'''
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

    res = mlops.register_unity_catalog_model(
        exp_root=tmp_path,
        catalog='prod_catalog',
        schema='landcover',
        model_name='sentinel_seg',
    )

    assert called_registry_uri == ['databricks-uc']
    assert registered_args['name'] == 'prod_catalog.landcover.sentinel_seg'
    assert registered_args['tags']['cluster_id'] == 'cluster-gpu-99'
    assert res.version == '3'


def test_register_model_decoupled_session_dpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: A standalone session output directory without exp_root.
    When: Registering model with session_dpath.
    Then: Resolve final checkpoint via `CheckpointCatalog`.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)

    session_dir = tmp_path / 'my_training_run'
    session_dir.mkdir()
    run = artifacts.SessionPaths.from_run_folder(
        session_dir
    ).init_pipeline_folders()
    best_ckpt = run.best_checkpoint('base_epoch')
    last_ckpt = run.last_checkpoint('base_epoch')
    best_ckpt.write_bytes(b'session_weights')
    last_ckpt.write_bytes(b'session_weights')
    cat = session.CheckpointCatalog(run=run)
    cat.record_save(
        'base', last_fpath=last_ckpt, epoch=1, best_fpath=best_ckpt, metric=0.8
    )
    cat.persist()

    result = mlops.register_unity_catalog_model(session_dpath=session_dir)
    assert result['status'] == 'SKIPPED_LOCAL'
    assert result['checkpoint_path'] == str(best_ckpt)


def test_register_model_ambiguous_legacy_checkpoints_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    '''
    Given: Session directory with multiple legacy `*_best.pt` and no catalog.
    When: Calling `register_unity_catalog_model`.
    Then: Raise `ArtifactError` instead of guessing by mtime.
    '''
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)

    session_dir = tmp_path / 'legacy_run'
    ckpt_dir = session_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / 'p1_epoch_best.pt').write_bytes(b'1')
    (ckpt_dir / 'p2_epoch_best.pt').write_bytes(b'2')

    with pytest.raises(artifacts.ArtifactError, match='Ambiguous legacy'):
        mlops.register_unity_catalog_model(session_dpath=session_dir)
