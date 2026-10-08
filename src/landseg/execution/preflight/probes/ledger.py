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
Ledger and lineage integrity diagnostic probes.

Inspects ETL run manifests (`harmonization_runs.json`, `ingestion_runs.json`)
and discovers pending batches or lineage discrepancies prior to execution.

Public APIs:
    - `past_runs`: Inspect past runs recorded in run manifest.
    - `pending_batches`: Check pending batches awaiting ingestion.
    - `canonical_pool_state`: Check block count in canonical pool.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- typing aliases
RunManifestCtrl = artifacts.Controller[dict[str, dict[str, str]]] # known type


# ----- public functions
def past_runs(
    manifest_path: str,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Inspect execution run history manifest.

    Args:
        manifest_path:
            file path to run history ledger JSON manifest.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            diagnostic probe record summarizing run status counts.
    '''
    try:
        manifest = RunManifestCtrl(manifest_path).fetch()
        if manifest is not None:
            total_run_n = len(manifest)
            success_run_n = len(
                [m for m in manifest.values() if m.get('status') == 'SUCCESS']
            )
            skipped_run_n = len(
                [m for m in manifest.values() if m.get('status') == 'SKIPPED']
            )
            failed_run_n = len(
                [m for m in manifest.values() if m.get('status') == 'FAILED']
            )

            return schema.ProbeResult(
                pid=pid or 'past_runs',
                category='Ledger',
                status=schema.ProbeStatus.PASS,
                message=(
                    f'Read run history manifest with {total_run_n} total runs '
                    f'with {success_run_n} success runs'
                ),
                details={
                    'path': manifest_path,
                    'total_runs': total_run_n,
                    'successful_runs': success_run_n,
                    'skipped_runs': skipped_run_n,
                    'failed_runs': failed_run_n,
                },
            )
        return schema.ProbeResult(
            pid=pid or 'past_runs',
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message='Read run history manifest with 0 runs',
            details={'path': manifest_path, 'runs': 'empty'},
        )

    except artifacts.ArtifactError as e:
        return schema.ProbeResult(
            pid=pid or 'past_runs',
            category='Ledger',
            status=schema.ProbeStatus.FAIL,
            message='Error reading run history manifest',
            details={'path': manifest_path, 'error': str(e)},
        )


def pending_harmonization(
    artifact_paths: artifacts.ArtifactPaths,
    root_config: configs.RootConfig
) -> schema.ProbeResult:
    '''Inspect if input source dataset is already harmonized.'''
    dataset_manifest_path = os.path.abspath(
        root_config.data.harmonization.dataset_manifest
    )

    manifest = RunManifestCtrl(
        artifact_paths.data_harmonization.runs_manifest
    ).fetch()
    if manifest is None:
        return schema.ProbeResult(
            pid='harmonized_dataset',
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message='Harmonization has not been run'
        )

    for v in manifest.values():
        if (
            v.get('source_dataset_manifest') == dataset_manifest_path and
            v.get('status') == 'SUCCESS'
        ):
            return schema.ProbeResult(
                    pid='harmonized_dataset',
                    category='Ledger',
                    status=schema.ProbeStatus.WARN,
                    message=(
                        f'Provided dataset already harmonized from run '
                        f'<{v.get('run_uid')}>'
                    ),
                    details={
                        'dataset_manifest_fpath': dataset_manifest_path,
                        'target_harmonization_run_uid': v.get('run_uid')
                    },
                )

    return schema.ProbeResult(
        pid='harmonized_dataset',
        category='Ledger',
        status=schema.ProbeStatus.PASS,
        message=f'Dataset from {dataset_manifest_path} not yet harmonized'
    )


def pending_ingestion(
    artifact_paths: artifacts.ArtifactPaths,
    pid: str | None = None,
    *,
    desire_pending: bool = True,
) -> schema.ProbeResult:
    '''
    Inspect pending harmonization batches awaiting ingestion.

    Args:
        artifact_paths:
            artifact path tree containing ETL run manifest paths.
        pid:
            optional probe identifier override.
        warn_if_pending:
            if True, emit WARN when batches are pending (for training).

    Returns:
        schema.ProbeResult:
            probe record indicating pending batch count and IDs.
    '''
    probe_id = pid or 'pending_ingestion'

    harmonize_manifest = artifact_paths.data_harmonization.runs_manifest
    ingest_manifest = artifact_paths.data_ingestion.runs_manifest

    harmonization_runs = RunManifestCtrl(harmonize_manifest).fetch()
    ingestion_runs = RunManifestCtrl(ingest_manifest).fetch()

    if not harmonization_runs:
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.FAIL,
            message='Unable to resolve pending batches',
            details={
                'harmonization_run_manifest': harmonize_manifest,
                'ingestion_run_manifest': ingest_manifest
            },
        )

    good_harmonization_runs = {
        k: v for k, v in harmonization_runs.items()
        if v.get('status') == 'SUCCESS'
    }

    effective_ingestion_runs = ingestion_runs or {}
    good_ingestion_runs = {
        k: v for k, v in effective_ingestion_runs.items()
        if v.get('status') == 'SUCCESS'
    }

    harmonization_uids = set(
        v.get('run_uid') for v in good_harmonization_runs.values()
    )
    harmonization_uids_in_ingestion = set(
        v.get('harmonization_run_uid') for v in good_ingestion_runs.values()
    )

    pending = harmonization_uids - harmonization_uids_in_ingestion

    match bool(pending), desire_pending:
        case True, True: status = schema.ProbeStatus.PASS
        case True, False: status = schema.ProbeStatus.WARN
        case False, True: status = schema.ProbeStatus.WARN
        case False, False: status = schema.ProbeStatus.PASS

    if pending:
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=status,
            message=f'{len(pending)} batches pending ingestion',
            details={'pending_run_ids': list(pending)},
        )

    return schema.ProbeResult(
        pid=probe_id,
        category='Ledger',
        status=status,
        message='Harmonization ledger up to date; nothing to ingest',
    )


def ingestion_pool_state(
    artifact_paths: artifacts.ArtifactPaths,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Inspect canonical block pool catalog state and block count.

    Args:
        catalog_path:
            file path to canonical blocks catalog JSON.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            probe record detailing existing blocks in canonical pool.
    '''
    probe_id = pid or 'ingested_blocks_pool'
    catalog_path = artifact_paths.data_ingestion.data_blocks.catalog

    if not os.path.exists(catalog_path):
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message='Canonical pool is empty (no blocks ingested yet)',
            details={'path': catalog_path, 'block_count': 0},
        )

    try:
        catalog = RunManifestCtrl(catalog_path).fetch()
        block_count = len(catalog) if catalog is not None else 0
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message=f'Ingestion pool contains {block_count} existing blocks',
            details={'path': catalog_path, 'block_count': block_count},
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.FAIL,
            message=f'Failed reading pool catalog: {err}',
            details={'path': catalog_path, 'error': str(err)},
        )


def prepared_blocks_state(
    artifact_paths: artifacts.ArtifactPaths,
    pid: str | None = None,
    *,
    desire_existing: bool = False,
) -> schema.ProbeResult:
    '''Inspect prepared blocks state.'''
    probe_id = pid or 'prepared_blocks_state'
    schema_path = artifact_paths.data_preparation.schema
    prep_schema = artifacts.Controller[dict](schema_path).fetch()

    match bool(prep_schema), desire_existing:
        case False, False:
            return schema.ProbeResult(
                pid=probe_id,
                category='Ledger',
                status=schema.ProbeStatus.PASS,
                message='Data preparation not yet run',
            )

        case False, True:
            return schema.ProbeResult(
                pid=probe_id,
                category='Ledger',
                status=schema.ProbeStatus.WARN,
                message='Data preparation schema not found; blocks not prepared',
            )

        case True, False:
            assert prep_schema is not None
            n_train = len(prep_schema.get('train_blocks', {}))
            n_val = len(prep_schema.get('val_blocks', {}))
            n_test = len(prep_schema.get('test_blocks', {}))
            return schema.ProbeResult(
                pid=probe_id,
                category='Ledger',
                status=schema.ProbeStatus.WARN,
                message=(
                    f'Found existing prepared blocks; '
                    f'train: {n_train} | val: {n_val} | test: {n_test}'
                ),
            )

        case True, True:
            assert prep_schema is not None
            n_train = len(prep_schema.get('train_blocks', {}))
            n_val = len(prep_schema.get('val_blocks', {}))
            n_test = len(prep_schema.get('test_blocks', {}))
            return schema.ProbeResult(
                pid=probe_id,
                category='Ledger',
                status=schema.ProbeStatus.PASS,
                message=(
                    f'Found {n_train} train / {n_val} val / {n_test} test '
                    'prepared blocks ready'
                ),
            )
