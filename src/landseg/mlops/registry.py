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
MLflow and Databricks Unity Catalog Model Registry integration.

Provides standardized registration of trained segmentation models to
enterprise Unity Catalog (`models:/<catalog>.<schema>.<model_name>`) with
governance metadata (manifest hashes, cluster ID, and git commit SHA).

Public APIs:
    - `register_unity_catalog_model`: Register checkpoint to Unity Catalog.
'''

# standard imports
import hashlib
import pathlib
import typing
# third-party imports
import mlflow
# local imports
import landseg.mlops.env as env_mod
import landseg.mlops.storage as storage_mod


# ----- public functions
def register_unity_catalog_model(
    exp_root: str | pathlib.Path,
    catalog: str = 'main',
    schema: str = 'landcover',
    model_name: str = 'landseg_unet',
    *,
    checkpoint_path: str | pathlib.Path | None = None,
    tags: dict[str, str] | None = None,
    dbutils: typing.Any = None,
) -> typing.Any:
    '''
    Register trained model checkpoint into Unity Catalog Model Registry.

    When running inside Databricks, sets registry URI to `databricks-uc`
    and registers the model under `models:/<catalog>.<schema>.<model_name>`
    alongside governance tags (manifest hash, cluster ID, git commit SHA).
    When executed outside Databricks or in headless CI, safely logs an
    inspection summary and returns metadata without raising exceptions.

    Args:
        exp_root:
            Active experiment root workspace directory.
        catalog:
            Unity Catalog name (default: 'main').
        schema:
            Unity Catalog schema/database name (default: 'landcover').
        model_name:
            Registered model name identifier (default: 'landseg_unet').
        checkpoint_path:
            Optional explicit path to the model checkpoint binary (.pt).
        tags:
            Optional custom metadata tags to attach to the model.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        typing.Any:
            `mlflow.entities.model_registry.ModelVersion` instance if registered
            in Databricks, or a summary metadata dictionary if executed locally.
    '''
    resolved_root = storage_mod.resolve_exp_root(exp_root, dbutils=dbutils)
    target_ckpt = _resolve_checkpoint(resolved_root, checkpoint_path)

    # Collect governance metadata
    manifest_hash = _probe_manifest_hash(resolved_root)
    git_sha = env_mod.get_git_commit()
    cluster_id = env_mod.get_cluster_id()
    runtime_version = env_mod.get_runtime_version()
    full_model_name = f'{catalog}.{schema}.{model_name}'

    governance_tags = {
        'framework': 'landseg',
        'cluster_id': cluster_id,
        'git_commit': git_sha,
        'manifest_sha256': manifest_hash if manifest_hash else 'unspecified',
        'databricks_runtime': runtime_version,
    }
    if tags:
        governance_tags.update(tags)

    # In local or CI non-Databricks environment: safe bypass
    if not env_mod.is_databricks():
        print(
            f'\n[UNITY CATALOG] Model registration bypassed (outside Databricks runtime):\n'
            f'- Target Model: models:/{full_model_name}\n'
            f'- Checkpoint: {target_ckpt}\n'
            f'- Manifest SHA256: {manifest_hash}\n'
            f'- Git Commit SHA: {git_sha}\n'
            f'- Status: Non-Databricks runtime detected, registration skipped safely.'
        )
        return {
            'status': 'SKIPPED_LOCAL',
            'full_model_name': full_model_name,
            'checkpoint_path': str(target_ckpt) if target_ckpt else None,
            'manifest_sha256': manifest_hash,
            'git_commit': git_sha,
            'tags': governance_tags,
        }

    # In Databricks runtime: publish to Unity Catalog Model Registry
    try:
        mlflow.set_registry_uri('databricks-uc')
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f'[UNITY CATALOG] Warning: Failed to set databricks-uc registry URI: {exc}')

    active_run = mlflow.active_run()
    close_run = False
    if active_run is None:
        run = mlflow.start_run(run_name='unity_catalog_registration')
        close_run = True
    else:
        run = active_run

    try:
        mlflow.set_tags(governance_tags)

        if target_ckpt and target_ckpt.is_file():
            mlflow.log_artifact(str(target_ckpt), artifact_path='model_checkpoint')
            model_uri = f'runs:/{run.info.run_id}/model_checkpoint'
        else:
            model_uri = f'runs:/{run.info.run_id}'

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=full_model_name,
            tags=governance_tags,
        )

        version_str = getattr(model_version, 'version', 'latest')
        print(
            f'\n[UNITY CATALOG] Successfully registered model:\n'
            f'- Model URI: models:/{full_model_name}\n'
            f'- Version: {version_str}\n'
            f'- Run ID: {run.info.run_id}\n'
            f'- Governance Tags: {governance_tags}'
        )
        return model_version

    finally:
        if close_run:
            mlflow.end_run()


# ----- private helpers
def _resolve_checkpoint(
    exp_root: pathlib.Path,
    explicit_path: str | pathlib.Path | None,
) -> pathlib.Path | None:
    '''Locate model checkpoint binary in explicit path or experiment folders.'''
    if explicit_path is not None:
        p = pathlib.Path(explicit_path)
        if p.is_file():
            return p

    search_dirs = [
        exp_root / 'results',
        exp_root / 'artifacts' / 'checkpoints',
        exp_root / 'artifacts' / 'session',
        exp_root / 'artifacts',
        exp_root,
    ]

    candidates: list[pathlib.Path] = []
    for sdir in search_dirs:
        if sdir.is_dir():
            candidates.extend(sdir.glob('**/*.pt'))

    valid = [p for p in candidates if 'interrupted' not in p.name and p.is_file()]
    if valid:
        # Prefer best checkpoint if available, otherwise most recently updated
        bests = [p for p in valid if 'best' in p.name]
        if bests:
            return max(bests, key=lambda p: p.stat().st_mtime)
        return max(valid, key=lambda p: p.stat().st_mtime)

    return None


def _probe_manifest_hash(exp_root: pathlib.Path) -> str | None:
    '''Compute or read SHA-256 hash of dataset manifest file.'''
    candidate_manifests = [
        exp_root / 'artifacts' / 'manifest.json',
        exp_root / 'artifacts' / 'manifest.yaml',
        exp_root / 'input' / 'manifest.json',
        exp_root / 'input' / 'manifest.yaml',
    ]
    for p in candidate_manifests:
        if p.is_file():
            try:
                hasher = hashlib.sha256()
                hasher.update(p.read_bytes())
                return hasher.hexdigest()
            except OSError:
                pass

    hash_files = list(exp_root.glob('**/_hash.json'))
    if hash_files:
        try:
            return hash_files[0].read_text(encoding='utf-8').strip()
        except OSError:
            pass

    return None
