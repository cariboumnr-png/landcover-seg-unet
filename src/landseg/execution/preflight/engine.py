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
Pre-flight suite execution engine.

Runs a target's declarative probe suite (see `catalog`) against an
`InspectionContext`, flattening list-valued checks and containing any
unexpected exception as a failing probe so a single misbehaving check
never aborts the whole inspection.

Public APIs:
    - `assert_target_ready`: Fail-fast gate for target execution.
    - `inspect_target`: Execute a target suite, returning a PreflightResult.
'''

# standard imports
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.catalog as catalog
import landseg.execution.preflight.context as context
import landseg.execution.preflight.reporter as reporter
import landseg.execution.preflight.schema as schema


# ----- public functions
def assert_target_ready(
    target: str,
    config: configs.RootConfig,
    *,
    artifact_paths: artifacts.ArtifactPaths | None = None,
) -> None:
    '''Assert execution readiness for a target.'''
    ctx = context.InspectionContext.from_config(
        config,
        artifact_paths=artifact_paths,
    )
    result = inspect_target(target, ctx)

    if not result.is_ready:
        failures = [
            f'{p.pid}: {p.message}'
            for p in result.probes
            if p.status == schema.ProbeStatus.FAIL
        ]
        raise RuntimeError(
            f'Preflight validation failed for "{target}":\n'
            + '\n'.join(f'  - {msg}' for msg in failures)
        )


def inspect_target(
    target: str,
    ctx: context.InspectionContext,
) -> schema.PreflightResult:
    '''
    Execute the diagnostic probe suite for a single execution target.

    Args:
        target:
            execution target identifier (e.g. `'data-ingest'`).
        ctx:
            pre-resolved inspection context (config + artifact paths).

    Returns:
        schema.PreflightResult:
            aggregated readiness evaluation and probe results.
    '''
    specs = catalog.suite_for(target)
    if not specs:
        return _blocked_unsupported(target)

    results: list[schema.ProbeResult] = []
    for spec in specs:
        results.extend(_run_spec(spec, ctx))

    all_ready = all(r.status != schema.ProbeStatus.FAIL for r in results)
    return schema.PreflightResult(
        target=target,
        status='READY' if all_ready else 'BLOCKED',
        probes=results,
    )


def run_preflight(
    root_config: configs.RootConfig,
    target: str | None = None,
    *,
    exp_root: pathlib.Path | None = None,
) -> schema.PreflightResult | list[schema.PreflightResult]:
    '''
    Dispatch pre-flight diagnostic checks for target pipeline(s).

    Args:
        root_config:
            hydra-composed root configuration.
        target:
            pipeline name, workflow name, or 'all'. If None, defaults
            to `root_config.execution.preflight_target` or 'all'.
        exp_root:
            optional experiment root directory override.

    Returns:
        schema.PreflightResult | list[schema.PreflightResult]:
            single result or list of results across all targets.
    '''
    cxt = context.InspectionContext.from_config(root_config)
    selected_target = (
        target
        or root_config.execution.preflight_target
        or 'all'
    )
    if selected_target in catalog.SUPPORTED_TARGETS:
        results: schema.PreflightResult | list[schema.PreflightResult]
        results = inspect_target(selected_target, cxt)

    elif selected_target == 'all':
        results = [inspect_target(t, cxt) for t in catalog.SUPPORTED_TARGETS]

    else:
        allowed = sorted(list(catalog.SUPPORTED_TARGETS) + ['all'])
        raise KeyError(
            f'Target "{selected_target}" not supported for preflight; '
            f'allowed: {allowed}'
        )

    # format and print terminal dashboard
    report_text = reporter.format_preflight_report(results)
    print(report_text)

    # export report artifact
    report_fp, _ = reporter.export_preflight_report(
        results,
        root_config,
        selected_target,
        exp_root=exp_root,
    )
    print(f'Preflight report saved to: {report_fp}')

    # strict mode enforcement
    if root_config.execution.preflight_strict_mode:
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
def _run_spec(
    spec: catalog.ProbeSpec,
    ctx: context.InspectionContext,
) -> list[schema.ProbeResult]:
    '''Run a single probe spec, flattening lists and trapping errors.'''
    try:
        outcome = spec.run(ctx)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return [
            schema.ProbeResult(
                pid=spec.pid,
                category=schema.ProbeCategory.EXECUTION,
                status=schema.ProbeStatus.FAIL,
                message=f'Probe raised unexpectedly: {err}',
                details={'error': str(err), 'error_type': type(err).__name__},
            )
        ]

    if isinstance(outcome, list):
        return outcome
    return [outcome]


def _blocked_unsupported(target: str) -> schema.PreflightResult:
    '''Build a BLOCKED result for a target lacking a registered suite.'''
    probe = schema.ProbeResult(
        pid='target_support',
        category=schema.ProbeCategory.EXECUTION,
        status=schema.ProbeStatus.FAIL,
        message=(f'Target "{target}" has no registered inspection suite'),
        details={'supported': list(catalog.SUPPORTED_TARGETS)},
    )
    return schema.PreflightResult(
        target=target,
        status='BLOCKED',
        probes=[probe],
    )
