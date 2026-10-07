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
Pre-flight execution engine and inspection dispatcher.

Coordinates non-destructive validation probe execution across pipeline
and workflow targets, handles reporting, and enforces readiness.

Public APIs:
    - `SUPPORTED_TARGETS`: execution targets supported by preflight engine.
    - `inspect_target`: Run diagnostic probe suite on an execution target.
    - `run_preflight`: Dispatch pre-flight checks for targets.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.prerequisites as prerequisites
import landseg.execution.preflight.probes as probes
import landseg.execution.preflight.reporter as reporter
import landseg.execution.preflight.schema as schema

SUPPORTED_TARGETS: tuple[str, ...] = (
    'world-grid',
    'data-harmonize',
    'data-ingest',
    'batch-ingest',
    'data-prepare',
    'model-train',
    'model-evaluate',
    'diagnose-overfit',
)


# ----- public functions
def inspect_target(
    target: str,
    root_config: configs.RootConfig,
) -> schema.PreflightResult:
    '''
    Run diagnostic probe suite on an execution target non-destructively.

    Args:
        target:
            execution target identifier or concrete pipeline runner instance.
        root_config:
            optional root configuration (inferred from pipeline if omitted).

    Returns:
        schema.PreflightResult:
            aggregated readiness evaluation and diagnostic probe results.
    '''
    results: list[schema.ProbeResult] = []
    artifact_paths = artifacts.ArtifactPaths.from_config(root_config)

    grid_paths = artifact_paths.world_grid
    harm_paths = artifact_paths.data_harmonization
    ingest_paths = artifact_paths.data_ingestion
    prep_paths = artifact_paths.data_preparation

    match target:
        case 'world-grid':
            results.extend([
                probes.dir_writable(grid_paths.root, 'world_grid_output'),
                probes.target_file_exists(grid_paths.report, 'world_grid_report'),
                probes.spatial_reference(root_config),
                probes.crs_info(root_config),
                probes.pixel_size(root_config),
                probes.grid_extent(root_config),
                probes.grid_origin(root_config),
                probes.grid_specs(root_config),
            ])

        case 'data-harmonize':
            results.extend([
                *_check_prerequisites(target, artifact_paths),
                probes.dir_writable(harm_paths.root, 'harmonization_output'),
                probes.raw_dataset(root_config),
                probes.past_runs(harm_paths.runs_manifest, 'past_harmonization_runs'),
                probes.pending_harmonization(artifact_paths, root_config),
            ])

        case 'data-ingest' | 'batch-ingest':
            results.extend([
                *_check_prerequisites(target, artifact_paths),
                probes.dir_writable(ingest_paths.root, 'ingestion_output'),
                probes.collision_policy(root_config),
                probes.past_runs(harm_paths.runs_manifest, 'past_harmonization_runs'),
                probes.past_runs(ingest_paths.runs_manifest, 'past_ingestion_runs'),
                probes.pending_ingestion(artifact_paths),
                probes.ingestion_pool_state(artifact_paths),
            ])

        case 'data-prepare':
            results.extend([
                *_check_prerequisites(target, artifact_paths),
                probes.dir_writable(prep_paths.root, 'preparation_output'),
                probes.target_file_exists(prep_paths.report, 'preparation_report'),
                probes.ingestion_pool_state(artifact_paths),
                probes.prepared_blocks_state(artifact_paths),
            ])

        case 'model-train':
            results.extend([
                *_check_prerequisites(target, artifact_paths),
                probes.dir_writable(artifact_paths.session_root, 'checkpoint_dir'),
                probes.model_body(root_config),
                probes.pending_ingestion(artifact_paths, desire_pending=False),
                probes.prepared_blocks_state(artifact_paths, desire_existing=True),
                *probes.hardware_info(root_config),
            ])

        case 'model-evaluate':
            results.extend([
                *_check_prerequisites(target, artifact_paths),
                probes.dir_writable(artifact_paths.session_root, 'eval_output'),
                probes.checkpoint_ready(root_config),
                probes.model_body(root_config),
                probes.eval_split(root_config),
                *probes.hardware_info(root_config),
            ])

        case 'diagnose-overfit':
            results.extend([
                probes.model_body(root_config),
                *probes.hardware_info(root_config),
            ])

        case _:
            results.append(
                schema.ProbeResult(
                    pid='target_support',
                    category='Execution',
                    status=schema.ProbeStatus.FAIL,
                    message=f'Target "{target}" has no registered preflight inspection suite',
                )
            )

    all_ready = all(p.status != schema.ProbeStatus.FAIL for p in results)
    status = 'READY' if all_ready else 'BLOCKED'

    return schema.PreflightResult(
        target=target,
        status=status,
        probes=results,
    )


def run_preflight(
    root_config: configs.RootConfig,
    target: str | None = None,
    *,
    exp_root: str | None = None,
) -> schema.PreflightResult | list[schema.PreflightResult]:
    '''
    Dispatch pre-flight diagnostic checks for target pipeline(s).

    Args:
        root_config:
            hydra-composed root configuration.
        target:
            pipeline name, workflow name, or 'all'. If None, defaults
            to `root_config.command.preflight.target`.
        exp_root:
            optional experiment root directory override.

    Returns:
        schema.PreflightResult | list[schema.PreflightResult]:
            single result or list of results across all targets.
    '''
    selected_target = target or root_config.command.preflight.target
    if selected_target in SUPPORTED_TARGETS:
        results: schema.PreflightResult | list[schema.PreflightResult]
        results = inspect_target(selected_target, root_config)

    elif selected_target == 'all':
        results = [inspect_target(t, root_config) for t in SUPPORTED_TARGETS]

    else:
        allowed = sorted(list(SUPPORTED_TARGETS) + ['all'])
        raise KeyError(
            f'Target "{selected_target}" not supported for preflight; '
            f'allowed: {allowed}'
        )

    # format and print terminal dashboard
    report_text = reporter.format_preflight_report(results)
    print(report_text)

    # export report artifact if configured
    if root_config.command.preflight.export_report:
        report_fp, _ = reporter.export_preflight_report(
            results,
            root_config,
            selected_target,
            exp_root=exp_root,
        )
        print(f'Preflight report saved to: {report_fp}')

    # strict mode enforcement
    if root_config.command.preflight.strict:
        result_list = (
            [results] if isinstance(results, schema.PreflightResult) else results
        )
        failures = [
            f'{r.target}: {", ".join(r.errors)}'
            for r in result_list
            if not r.is_ready
        ]
        if failures:
            raise RuntimeError(
                f'Strict preflight validation failed: {"; ".join(failures)}'
            )

    return results


# ----- private helpers
def _check_prerequisites(
    target: str,
    artifact_paths: artifacts.ArtifactPaths,
) -> list[schema.ProbeResult]:
    '''Check prerequisite artifacts for target pipeline.'''
    checks = prerequisites.check_target_prerequisites(target, artifact_paths)
    return [check.to_probe_result() for check in checks]
