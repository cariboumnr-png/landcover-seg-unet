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
Upstream lineage diagnostic checks.

Verifies that upstream pipeline targets have produced the reports,
manifests, and artifacts a downstream target depends on, and inspects
ETL run ledgers for pending batches and run history.

Every check returns a `ProbeResult` and performs no side effects.

Public APIs:
    - `upstream_report`: Verify an upstream target report succeeded.
    - `upstream_manifest`: Verify an upstream runs manifest has runs.
    - `required_artifacts`: Verify required artifact files exist.
    - `run_history`: Summarize a run manifest's status counts.
    - `pending_harmonization`: Detect already-harmonized input dataset.
    - `pending_ingestion`: Detect harmonization batches awaiting ingest.
'''

# standard imports
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema


# ----- constants
_CATEGORY = schema.ProbeCategory.LINEAGE


# ----- public functions
def upstream_report(
    report_fpath: pathlib.Path,
    upstream_target: str,
    pid: str = 'upstream_report',
    *,
    expected_status: str = 'SUCCESS',
) -> schema.ProbeResult:
    '''Verify an upstream pipeline report exists with expected status.'''
    if not report_fpath.is_file():
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" report not found at: '
                f'{report_fpath}'
            ),
            details={
                'upstream_target': upstream_target,
                'report_fpath': str(report_fpath),
                'expected_status': expected_status,
            },
        )

    try:
        data = artifacts.Controller[dict](report_fpath).fetch()
    except artifacts.ArtifactError as err:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" report is unreadable '
                f'at: {report_fpath} ({err})'
            ),
            details={
                'upstream_target': upstream_target,
                'report_fpath': str(report_fpath),
                'error': str(err),
            },
        )

    status_val = data.get('status')
    if status_val != expected_status:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" status is '
                f'"{status_val}", not "{expected_status}". '
                f'Please re-run "{upstream_target}" successfully first.'
            ),
            details={
                'upstream_target': upstream_target,
                'report_fpath': str(report_fpath),
                'actual_status': status_val,
                'expected_status': expected_status,
            },
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=f'Upstream "{upstream_target}" prerequisites verified.',
        details={
            'upstream_target': upstream_target,
            'report_fpath': str(report_fpath),
            'status': status_val,
        },
    )


def upstream_manifest(
    manifest_fpath: pathlib.Path,
    upstream_target: str,
    pid: str = 'upstream_manifest',
    *,
    min_success_runs: int = 1,
) -> schema.ProbeResult:
    '''Verify an upstream runs manifest has enough successful runs.'''
    if not manifest_fpath.is_file():
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" runs manifest '
                f'not found at: {manifest_fpath}'
            ),
            details={
                'upstream_target': upstream_target,
                'manifest_fpath': str(manifest_fpath),
            },
        )

    try:
        data = artifacts.Controller[dict](manifest_fpath).fetch()
    except artifacts.ArtifactError as err:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" runs manifest is '
                f'unreadable at: {manifest_fpath} ({err})'
            ),
            details={
                'upstream_target': upstream_target,
                'manifest_fpath': str(manifest_fpath),
                'error': str(err),
            },
        )

    records = [v for v in (data or {}).values() if isinstance(v, dict)]
    success_count = sum(1 for r in records if r.get('status') == 'SUCCESS')
    if success_count < min_success_runs:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" has no successful '
                f'runs recorded in {manifest_fpath}.'
            ),
            details={
                'upstream_target': upstream_target,
                'manifest_fpath': str(manifest_fpath),
                'total_runs': len(records),
                'successful_runs': success_count,
            },
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=(
            f'Upstream pipeline "{upstream_target}" manifest verified '
            f'({success_count} successful run(s)).'
        ),
        details={
            'upstream_target': upstream_target,
            'manifest_fpath': str(manifest_fpath),
            'successful_runs': success_count,
        },
    )


def required_artifacts(
    file_paths: list[pathlib.Path],
    upstream_target: str,
    pid: str = 'required_artifacts',
) -> schema.ProbeResult:
    '''Verify that required upstream artifact files exist on disk.'''
    missing = [str(fp) for fp in file_paths if not fp.exists()]
    if missing:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Upstream pipeline "{upstream_target}" has not produced '
                f'required artifacts. Missing: {", ".join(missing)}'
            ),
            details={
                'upstream_target': upstream_target,
                'missing_files': missing,
            },
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=f'Required artifacts verified for "{upstream_target}".',
        details={
            'upstream_target': upstream_target,
            'checked_files': [str(fp) for fp in file_paths],
        },
    )


def run_history(
    manifest_fp: pathlib.Path,
    pid: str,
) -> schema.ProbeResult:
    '''Summarize status counts recorded in a run history manifest.'''
    mfst_str = str(manifest_fp.resolve())
    try:
        mfst = artifacts.Controller[dict](manifest_fp).fetch(missing_ok=True)
    except artifacts.ArtifactError as err:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message='Error reading run history manifest',
            details={'path': mfst_str, 'error': str(err)},
        )

    if not mfst:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.PASS,
            message='Read run history manifest with 0 runs',
            details={'path': mfst_str, 'runs': 'empty'},
        )

    def _count(status: str) -> int:
        return sum(1 for m in mfst.values() if m.get('status') == status)

    total = len(mfst)
    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=(
            f'Read run history manifest with {total} total runs '
            f'with {_count("SUCCESS")} success runs'
        ),
        details={
            'path': mfst_str,
            'total_runs': total,
            'successful_runs': _count('SUCCESS'),
            'skipped_runs': _count('SKIPPED'),
            'failed_runs': _count('FAILED'),
        },
    )


def pending_harmonization(
    ctx: context.InspectionContext,
    pid: str = 'harmonized_dataset',
) -> schema.ProbeResult:
    '''Detect whether the configured dataset is already harmonized.'''
    paths = ctx.paths
    harmonize_mfst = paths.data_harmonization.runs_manifest
    try:
        harmonization_runs = artifacts.Controller[dict](harmonize_mfst).fetch()
    except artifacts.ArtifactError:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.PASS,
            message='Harmonization has not been run',
        )

    dataset_manifest = ctx.config.data.harmonization.dataset_manifest
    manifest_path_str = str(pathlib.Path(dataset_manifest).resolve())
    for record in (harmonization_runs or {}).values():
        if (
            record.get('source_dataset_manifest') == manifest_path_str and
            record.get('status') == 'SUCCESS'
        ):
            return schema.ProbeResult(
                pid=pid,
                category=_CATEGORY,
                status=schema.ProbeStatus.WARN,
                message=(
                    f'Provided dataset already harmonized from run '
                    f'<{record.get("run_uid")}>'
                ),
                details={
                    'dataset_manifest_fpath': manifest_path_str,
                    'target_harmonization_run_uid': record.get('run_uid'),
                },
            )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=f'Dataset from {manifest_path_str} not yet harmonized',
    )


def pending_ingestion(
    ctx: context.InspectionContext,
    pid: str = 'pending_ingestion',
    *,
    desire_pending: bool = True,
) -> schema.ProbeResult:
    '''Detect harmonization batches awaiting ingestion.'''
    paths = ctx.paths
    harmonize_mfst = paths.data_harmonization.runs_manifest
    ingest_mfst = paths.data_ingestion.runs_manifest
    Ctrl = artifacts.Controller[dict]

    try:
        harmonization_runs = Ctrl(harmonize_mfst).fetch()
    except artifacts.ArtifactError:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message='Unable to resolve pending batches',
            details={
                'harmonization_run_manifest': str(harmonize_mfst.resolve()),
                'ingestion_run_manifest': str(ingest_mfst.resolve()),
            },
        )

    harmonization_uids = {
        v.get('run_uid') for v in harmonization_runs.values()
        if v.get('status') == 'SUCCESS'
    }

    try:
        ingestion_runs = Ctrl(ingest_mfst).fetch()
        ingested_uids = {
            v.get('harmonization_run_uid') for v in ingestion_runs.values()
            if v.get('status') == 'SUCCESS'
        }
    except artifacts.ArtifactError:
        ingested_uids = set()

    pending = harmonization_uids - ingested_uids

    match bool(pending), desire_pending:
        case True, True: status = schema.ProbeStatus.PASS
        case True, False: status = schema.ProbeStatus.WARN
        case False, True: status = schema.ProbeStatus.WARN
        case False, False: status = schema.ProbeStatus.PASS

    if pending:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=status,
            message=f'{len(pending)} batches pending ingestion',
            details={'pending_run_ids': sorted(pending)},
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=status,
        message='Harmonization ledger up to date; nothing to ingest',
    )
