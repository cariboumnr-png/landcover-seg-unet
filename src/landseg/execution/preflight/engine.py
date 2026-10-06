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
    - `inspect_target`: Run diagnostic probe suite on an execution target.
    - `run_preflight`: Dispatch pre-flight checks for targets.
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.execution.pipelines.base as base
import landseg.execution.preflight.probes as probes
import landseg.execution.preflight.reporter as reporter
import landseg.execution.preflight.schema as schema


# ----- public functions
def inspect_target(
    target: str | base.BasePipeline,
    root_config: configs.RootConfig | None = None,
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
    if isinstance(target, base.BasePipeline):
        target_name = target.pipeline_name
        eff_config = root_config or target.config
    else:
        target_name = target
        eff_config = root_config or configs.RootConfig()

    probe_results: list[schema.ProbeResult] = []
    telemetry: dict[str, typing.Any] = {}

    check_gpu = eff_config.command.preflight.check_gpu

    probe_results.extend(
        probes.probe_hardware(
            check_gpu=check_gpu,
            telemetry=telemetry,
            root_config=eff_config,
        )
    )
    probe_results.extend(probes.probe_storage(target, root_config=eff_config))
    probe_results.extend(probes.probe_lineage(target, root_config=eff_config))
    probe_results.extend(probes.probe_spatial(target_name, eff_config))
    probe_results.extend(probes.probe_ledger(target_name, eff_config))
    probe_results.extend(probes.probe_model(target_name, eff_config))
    probe_results.extend(probes.probe_study(target_name, eff_config))

    all_ready = all(p.status != schema.ProbeStatus.FAIL for p in probe_results)
    status = 'READY' if all_ready else 'BLOCKED'

    return schema.PreflightResult(
        target=target_name,
        status=status,
        probes=probe_results,
        telemetry=telemetry,
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
    supported_targets = [
        'world-grid',
        'data-harmonize',
        'data-ingest',
        'data-prepare',
        'model-train',
        'model-evaluate',
        'batch-ingest',
        'diagnose-overfit',
        'study-sweep',
        'study-analysis',
    ]
    target_runners: dict[str, typing.Callable[[], schema.PreflightResult]] = {
        name: (lambda n=name: inspect_target(n, root_config))
        for name in supported_targets
    }

    if selected_target in target_runners:
        results: schema.PreflightResult | list[schema.PreflightResult]
        results = target_runners[selected_target]()
    elif selected_target == 'all':
        results = [runner_fn() for runner_fn in target_runners.values()]
    else:
        allowed = sorted(list(target_runners.keys()) + ['all'])
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
