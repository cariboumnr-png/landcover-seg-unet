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
import landseg.execution.preflight.schema as schema
import landseg.geopipe.ingest as ingest


# ----- typing aliases
DictCtrl = artifacts.Controller[dict]


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
        manifest = DictCtrl(manifest_path).fetch()
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


def pending_batches(
    harmonization_manifest: str,
    ingestion_manifest: str,
    *,
    target: int | str | None = None,
    rebuild: bool = False,
    warn_if_pending: bool = False,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Inspect pending harmonization batches awaiting ingestion.

    Args:
        harmonization_manifest:
            file path to harmonization runs manifest JSON.
        ingestion_manifest:
            file path to ingestion runs manifest JSON.
        target:
            optional harmonization run target identifier or index.
        rebuild:
            whether already ingested runs are considered for rebuild.
        warn_if_pending:
            if True, emit WARN when batches are pending (for training).
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            probe record indicating pending batch count and IDs.
    '''
    probe_id = pid or (
        'pending_data_warning' if warn_if_pending else 'pending_batch_queue'
    )
    try:
        pending = ingest.resolve_pending_ingestion_batches(
            harmonization_manifest,
            ingestion_manifest,
            target=target,
            rebuild=rebuild,
        )
        run_ids = [p['run_id'] for p in pending]

        if warn_if_pending:
            if pending:
                return schema.ProbeResult(
                    pid=probe_id,
                    category='Ledger',
                    status=schema.ProbeStatus.WARN,
                    message=f'{len(pending)} batches pending in harmonization',
                    details={'pending_run_ids': run_ids},
                )
            return schema.ProbeResult(
                pid=probe_id,
                category='Ledger',
                status=schema.ProbeStatus.PASS,
                message='Harmonization ledger up to date; no pending batches',
            )

        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message=(
                f'{len(pending)} batches pending in queue'
                if pending
                else 'Batch queue is empty; pool is up to date'
            ),
            details={'pending_run_ids': run_ids},
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.FAIL,
            message=f'Failed resolving pending batches: {err}',
            details={'error': str(err)},
        )


def canonical_pool_state(
    catalog_path: str,
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
    probe_id = pid or 'canonical_pool_state'
    if not os.path.exists(catalog_path):
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message='Canonical pool is empty (no blocks ingested yet)',
            details={'path': catalog_path, 'block_count': 0},
        )

    try:
        catalog = DictCtrl(catalog_path).fetch()
        block_count = len(catalog) if catalog is not None else 0
        return schema.ProbeResult(
            pid=probe_id,
            category='Ledger',
            status=schema.ProbeStatus.PASS,
            message=f'Pool contains {block_count} existing blocks',
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
