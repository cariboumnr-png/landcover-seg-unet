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
Execution prerequisite validation and lineage verification.

Provides parameterized diagnostic checks and execution gates to evaluate
pipeline upstream readiness non-destructively before execution.

Public APIs:
    - `PrerequisiteCheck`: Evaluation record with fail-fast capability.
    - `check_report_status`: Check upstream JSON report success.
    - `check_files_exist`: Check required canonical files on disk.
    - `check_manifest_runs`: Check ledger runs manifest for success.
    - `check_target_prerequisites`: Evaluate prerequisites for target.
    - `assert_target_prerequisites`: Fail-fast gate for target execution.
'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PrerequisiteCheck:
    '''Evaluation record representing upstream prerequisite status.'''

    target: str
    upstream_target: str | None
    is_valid: bool
    message: str
    details: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def passed(
        cls,
        target: str,
        upstream_target: str | None,
        message: str,
        **details: typing.Any,
    ) -> 'PrerequisiteCheck':
        '''Construct a passing prerequisite check record.'''
        return cls(
            target=target,
            upstream_target=upstream_target,
            is_valid=True,
            message=message,
            details=details,
        )

    @classmethod
    def failed(
        cls,
        target: str,
        upstream_target: str | None,
        message: str,
        **details: typing.Any,
    ) -> 'PrerequisiteCheck':
        '''Construct a failing prerequisite check record.'''
        return cls(
            target=target,
            upstream_target=upstream_target,
            is_valid=False,
            message=message,
            details=details,
        )

    def raise_if_failed(self) -> None:
        '''
        Raise runtime error if prerequisite condition failed.

        Raises:
            RuntimeError:
                if prerequisite status is not valid.
        '''
        if not self.is_valid:
            raise RuntimeError(self.message)

    def to_probe_result(
        self,
        probe_id: str = 'pipeline_prerequisites',
        category: str = 'lineage',
    ) -> schema.ProbeResult:
        '''
        Convert prerequisite evaluation into preflight probe result.

        Args:
            probe_id:
                unique identifier for the diagnostic probe.
            category:
                diagnostic probe category label.

        Returns:
            schema.ProbeResult:
                structured probe record with pass or fail status.
        '''
        return schema.ProbeResult(
            probe_id=probe_id,
            category=category,
            status=(
                schema.ProbeStatus.PASS
                if self.is_valid
                else schema.ProbeStatus.FAIL
            ),
            message=self.message,
            details=self.details,
        )


# ----- public functions
def check_report_status(
    target: str,
    upstream_target: str,
    report_fpath: str,
    expected_status: str = 'SUCCESS',
) -> PrerequisiteCheck:
    '''
    Validate that upstream report exists and has expected status.

    Args:
        target:
            current execution target identifier.
        upstream_target:
            name of upstream pipeline producing report.
        report_fpath:
            file path to upstream report JSON.
        expected_status:
            expected terminal status string in report.

    Returns:
        PrerequisiteCheck:
            evaluation result containing validity and message.
    '''
    if not os.path.isfile(report_fpath):
        message = (
            f'Upstream pipeline "{upstream_target}" report not found at: '
            f'{report_fpath}'
        )
        return PrerequisiteCheck.failed(
            target,
            upstream_target,
            message,
            report_fpath=report_fpath,
            expected_status=expected_status,
        )

    try:
        report_ctrl = artifacts.Controller[dict].load_json_or_fail(report_fpath)
        data = report_ctrl.fetch()
        status_val = data.get('status')
        if status_val != expected_status:
            message = (
                f'Upstream pipeline "{upstream_target}" status is '
                f'"{status_val}", not "{expected_status}". '
                f'Please re-run "{upstream_target}" successfully first.'
            )
            return PrerequisiteCheck.failed(
                target,
                upstream_target,
                message,
                report_fpath=report_fpath,
                actual_status=status_val,
                expected_status=expected_status,
            )

        message = f'Upstream "{upstream_target}" prerequisites verified.'
        return PrerequisiteCheck.passed(
            target,
            upstream_target,
            message,
            report_fpath=report_fpath,
            status=status_val,
        )

    except artifacts.ArtifactError as err:
        message = (
            f'Upstream pipeline "{upstream_target}" report is missing '
            f'or unreadable at: {report_fpath} ({err})'
        )
        return PrerequisiteCheck.failed(
            target,
            upstream_target,
            message,
            report_fpath=report_fpath,
            expected_status=expected_status,
            error=str(err),
        )


def check_files_exist(
    target: str,
    upstream_target: str,
    file_paths: list[str],
) -> PrerequisiteCheck:
    '''
    Validate that required upstream artifact files exist on disk.

    Args:
        target:
            current execution target identifier.
        upstream_target:
            name of upstream pipeline producing artifacts.
        file_paths:
            list of file paths required on disk.

    Returns:
        PrerequisiteCheck:
            evaluation result containing validity and message.
    '''
    missing = [fp for fp in file_paths if not os.path.exists(fp)]

    if missing:
        message = (
            f'Upstream pipeline "{upstream_target}" has not produced '
            f'required artifacts. Missing: {", ".join(missing)}'
        )
        return PrerequisiteCheck.failed(
            target,
            upstream_target,
            message,
            missing_files=missing,
        )

    message = f'Required artifacts verified for "{upstream_target}".'
    return PrerequisiteCheck.passed(
        target,
        upstream_target,
        message,
        checked_files=file_paths,
    )


def check_manifest_runs(
    target: str,
    upstream_target: str,
    manifest_fpath: str,
    min_success_runs: int = 1,
) -> PrerequisiteCheck:
    '''
    Validate that upstream run manifest exists and has completed runs.

    Args:
        target:
            current execution target identifier.
        upstream_target:
            name of upstream pipeline producing manifest.
        manifest_fpath:
            file path to runs manifest JSON.
        min_success_runs:
            minimum number of successful run records required.

    Returns:
        PrerequisiteCheck:
            evaluation result containing validity and message.
    '''
    if not os.path.isfile(manifest_fpath):
        message = (
            f'Upstream pipeline "{upstream_target}" runs manifest '
            f'not found at: {manifest_fpath}'
        )
        return PrerequisiteCheck.failed(
            target,
            upstream_target,
            message,
            manifest_fpath=manifest_fpath,
        )

    try:
        manifest_ctrl = artifacts.Controller.load_json_or_fail(manifest_fpath)
        data = manifest_ctrl.fetch()
        records: list[dict[str, typing.Any]] = []

        if isinstance(data, dict):
            records = [v for v in data.values() if isinstance(v, dict)]
        elif isinstance(data, list):
            records = [v for v in data if isinstance(v, dict)]

        success_count = sum(
            1 for r in records if r.get('status') == 'SUCCESS'
        )

        if success_count < min_success_runs:
            message = (
                f'Upstream pipeline "{upstream_target}" has no successful '
                f'runs recorded in {manifest_fpath}.'
            )
            return PrerequisiteCheck.failed(
                target,
                upstream_target,
                message,
                manifest_fpath=manifest_fpath,
                total_runs=len(records),
                successful_runs=success_count,
            )

        message = (
            f'Upstream pipeline "{upstream_target}" manifest verified '
            f'({success_count} successful run(s)).'
        )
        return PrerequisiteCheck.passed(
            target,
            upstream_target,
            message,
            manifest_fpath=manifest_fpath,
            successful_runs=success_count,
        )

    except artifacts.ArtifactError as err:
        message = (
            f'Upstream pipeline "{upstream_target}" runs manifest is '
            f'missing or unreadable at: {manifest_fpath} ({err})'
        )
        return PrerequisiteCheck.failed(
            target,
            upstream_target,
            message,
            manifest_fpath=manifest_fpath,
            error=str(err),
        )


def check_target_prerequisites(
    target: str,
    artifact_paths: artifacts.ArtifactPaths,
    root_config: configs.RootConfig | None = None,
) -> list[PrerequisiteCheck]:
    '''
    Evaluate all upstream prerequisite checks for an execution target.

    Args:
        target:
            target execution pipeline or workflow identifier.
        artifact_paths:
            resolved experiment artifact paths instance.
        root_config:
            optional root configuration for parameter-dependent checks.

    Returns:
        list[PrerequisiteCheck]:
            list of prerequisite check results for the target.
    '''
    del root_config
    checks: list[PrerequisiteCheck] = []

    if target == 'world-grid':
        return checks

    if target == 'data-harmonize':
        checks.append(
            check_report_status(
                target='data-harmonize',
                upstream_target='world-grid',
                report_fpath=artifact_paths.world_grid.report,
            )
        )

    elif target == 'data-ingest':
        checks.append(
            check_manifest_runs(
                target='data-ingest',
                upstream_target='data-harmonize',
                manifest_fpath=(
                    artifact_paths.data_harmonization.runs_manifest
                ),
            )
        )

    elif target == 'data-prepare':
        checks.append(
            check_files_exist(
                target='data-prepare',
                upstream_target='data-ingest',
                file_paths=[
                    artifact_paths.data_ingestion.data_blocks.catalog,
                    artifact_paths.data_ingestion.data_blocks.schema,
                ],
            )
        )
        checks.append(
            check_manifest_runs(
                target='data-prepare',
                upstream_target='data-ingest',
                manifest_fpath=(
                    artifact_paths.data_ingestion.runs_manifest
                ),
            )
        )

    elif target in {'model-train', 'model-evaluate'}:
        checks.append(
            check_report_status(
                target=target,
                upstream_target='data-prepare',
                report_fpath=artifact_paths.data_preparation.report,
            )
        )

    return checks


def assert_target_prerequisites(
    target: str,
    artifact_paths: artifacts.ArtifactPaths,
    root_config: configs.RootConfig | None = None,
) -> None:
    '''
    Assert that all prerequisites pass for a target, raising on failure.

    Args:
        target:
            target execution pipeline or workflow identifier.
        artifact_paths:
            resolved experiment artifact paths instance.
        root_config:
            optional root configuration for parameter-dependent checks.

    Raises:
        RuntimeError:
            if any prerequisite condition fails.
    '''
    checks = check_target_prerequisites(target, artifact_paths, root_config)
    for check in checks:
        check.raise_if_failed()
