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
Pre-flight execution engine and pipeline inspection dispatcher.

Coordinates non-destructive validation probe execution across pipeline
targets, handles reporting, and enforces execution readiness.

Public APIs:
    - `inspect_pipeline`: Run diagnostic probe suite on a pipeline.
    - `run_preflight`: Dispatch pre-flight checks for targets.
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.pipelines.base as base
import landseg.execution.preflight.probes as probes
import landseg.execution.preflight.reporter as reporter
import landseg.execution.preflight.schema as schema


# ----- public functions
def inspect_pipeline(
    pipeline: base.BasePipeline,
    root_config: configs.RootConfig | None = None,
) -> schema.PreflightResult:
    '''
    Run pre-flight validation probes on a pipeline non-destructively.

    Args:
        pipeline:
            concrete pipeline runner instance to inspect.
        root_config:
            optional root configuration for hardware and runtime context.

    Returns:
        schema.PreflightResult:
            aggregated readiness evaluation and diagnostic probe results.
    '''
    probe_results: list[schema.ProbeResult] = []
    telemetry: dict[str, typing.Any] = {}

    check_gpu = (
        root_config.command.preflight.check_gpu
        if root_config is not None
        else True
    )

    probe_results.extend(
        probes.probe_hardware(check_gpu=check_gpu, telemetry=telemetry)
    )
    probe_results.extend(probes.probe_storage(pipeline))
    probe_results.extend(probes.probe_lineage(pipeline))

    all_ready = all(p.status != schema.ProbeStatus.FAIL for p in probe_results)
    status = 'READY' if all_ready else 'BLOCKED'

    return schema.PreflightResult(
        target=pipeline.pipeline_name,
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
            pipeline name or 'all'. If None, defaults to the target
            specified in `root_config.command.preflight.target`.
        exp_root:
            optional experiment root directory override.

    Returns:
        schema.PreflightResult | list[schema.PreflightResult]:
            single result or list of results across all targets.
    '''
    selected_target = target or root_config.command.preflight.target
    pipeline_runners: dict[
        str, typing.Callable[[], schema.PreflightResult]
    ] = {
        'world-grid': lambda: (
            inspect_pipeline(
                pipelines.WorldGridGeneration(root_config), root_config
            )
        ),
        'data-harmonize': lambda: (
            inspect_pipeline(
                pipelines.DataHarmonization(root_config), root_config
            )
        ),
        'data-ingest': lambda: (
            inspect_pipeline(
                pipelines.DataIngestion(root_config), root_config
            )
        ),
        'data-prepare': lambda: (
            inspect_pipeline(
                pipelines.DataPreparation(root_config), root_config
            )
        ),
        'model-train': lambda: (
            inspect_pipeline(
                pipelines.ModelTraining(root_config), root_config
            )
        ),
        'model-evaluate': lambda: (
            inspect_pipeline(
                pipelines.ModelEvaluation(root_config), root_config
            )
        ),
    }

    if selected_target in pipeline_runners:
        results: schema.PreflightResult | list[
            schema.PreflightResult
        ] = pipeline_runners[selected_target]()
    elif selected_target == 'all':
        results = [runner_fn() for runner_fn in pipeline_runners.values()]
    else:
        allowed = sorted(list(pipeline_runners.keys()) + ['all'])
        raise KeyError(
            f'Target "{selected_target}" not supported for pipeline preflight; '
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
