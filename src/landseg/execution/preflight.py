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
Pre-flight non-destructive validation engine for pipeline execution.

Provides diagnostic probe data structures and standalone inspection
routines to verify prerequisites without committing compute or
modifying disk state.

Public APIs:
    - `ProbeStatus`: status enumeration for diagnostic probes.
    - `ProbeResult`: immutable result record for a single probe.
    - `PreflightResult`: aggregated pre-flight report with helpers.
    - `inspect_pipeline`: run validation probes on a pipeline.
    - `run_preflight`: execute pre-flight checks for targets.
'''

# standard imports
import dataclasses
import enum
import typing
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.pipelines.base as base


# ----- public types
class ProbeStatus(enum.StrEnum):
    '''Diagnostic probe execution status.'''

    PASS = 'PASS'
    WARN = 'WARN'
    FAIL = 'FAIL'
    SKIP = 'SKIP'


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class ProbeResult:
    '''Single diagnostic probe evaluation result.'''

    probe_id: str
    category: str
    status: ProbeStatus
    message: str
    details: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PreflightResult:
    '''Aggregated pre-flight inspection result for a pipeline.'''

    target: str
    status: str
    probes: list[ProbeResult] = dataclasses.field(default_factory=list)
    telemetry: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        '''Return True if no probes have failed.'''
        return all(p.status != ProbeStatus.FAIL for p in self.probes)

    @property
    def errors(self) -> list[str]:
        '''Return error messages from failed probes.'''
        return [p.message for p in self.probes if p.status == ProbeStatus.FAIL]

    @property
    def warnings(self) -> list[str]:
        '''Return warning messages from warning probes.'''
        return [p.message for p in self.probes if p.status == ProbeStatus.WARN]

    def as_dict(self) -> dict[str, typing.Any]:
        '''Return dictionary representation for report serialization.'''
        return {
            'target': self.target,
            'status': self.status,
            'is_ready': self.is_ready,
            'probes': [
                {
                    'probe_id': p.probe_id,
                    'category': p.category,
                    'status': p.status.value,
                    'message': p.message,
                    'details': p.details,
                }
                for p in self.probes
            ],
            'errors': self.errors,
            'warnings': self.warnings,
            'telemetry': self.telemetry,
        }


# ----- public functions
def inspect_pipeline(pipeline: base.BasePipeline) -> PreflightResult:
    '''
    Run pre-flight validation probes on a pipeline non-destructively.

    Evaluates pipeline prerequisites without committing compute or
    modifying disk state, capturing validation outcomes into
    structured diagnostic probes.

    Args:
        pipeline:
            concrete pipeline runner instance to inspect.

    Returns:
        PreflightResult:
            aggregated readiness evaluation and diagnostic probe results.
    '''
    probes: list[ProbeResult] = []
    try:
        pipeline.validate()
        probes.append(
            ProbeResult(
                probe_id='pipeline_prerequisites',
                category='lineage',
                status=ProbeStatus.PASS,
                message=(
                    f'Prerequisites verified for pipeline '
                    f'"{pipeline.pipeline_name}".'
                ),
            )
        )
        status = 'READY'
    except Exception as err:  # pylint: disable=broad-exception-caught
        probes.append(
            ProbeResult(
                probe_id='pipeline_prerequisites',
                category='lineage',
                status=ProbeStatus.FAIL,
                message=str(err),
                details={'error_type': type(err).__name__},
            )
        )
        status = 'BLOCKED'

    return PreflightResult(
        target=pipeline.pipeline_name,
        status=status,
        probes=probes,
    )


def run_preflight(
    root_config: configs.RootConfig,
    target: str | None = None
) -> PreflightResult | list[PreflightResult]:
    '''
    Dispatch pre-flight diagnostic checks for target pipeline(s).

    Args:
        root_config:
            hydra-composed root configuration.
        target:
            pipeline name or 'all'. If None, defaults to the target
            specified in `root_config.command.preflight.target`.

    Returns:
        PreflightResult | list[PreflightResult]:
            single result or list of results across all targets.
    '''
    selected_target = target or root_config.command.preflight.target
    pipeline_runners: dict[str, typing.Callable[[], PreflightResult]] = {
        'world-grid': lambda: (
            inspect_pipeline(pipelines.WorldGridGeneration(root_config))
        ),
        'data-harmonize': lambda: (
            inspect_pipeline(pipelines.DataHarmonization(root_config))
        ),
        'data-ingest': lambda: (
            inspect_pipeline(pipelines.DataIngestion(root_config))
        ),
        'data-prepare': lambda: (
            inspect_pipeline(pipelines.DataPreparation(root_config))
        ),
        'model-train': lambda: (
            inspect_pipeline(pipelines.ModelTraining(root_config))
        ),
        'model-evaluate': lambda: (
            inspect_pipeline(pipelines.ModelEvaluation(root_config))
        ),
    }

    if selected_target in pipeline_runners:
        return pipeline_runners[selected_target]()

    if selected_target == 'all':
        return [runner_fn() for runner_fn in pipeline_runners.values()]

    allowed = sorted(list(pipeline_runners.keys()) + ['all'])
    raise KeyError(
        f'Target "{selected_target}" not supported for pipeline preflight; '
        f'allowed: {allowed}'
    )
