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
Declarative pre-flight target suites.

Defines, per execution target, the ordered set of checks that must run
to assess readiness. This table is the single source of "what readiness
means" for each target and is consumed both by the inspection engine
(informational) and by execution gating (fail-fast).

Public APIs:
    - `SUPPORTED_TARGETS`: Execution targets with a registered suite.
    - `ProbeSpec`: A single named check bound to a callable.
    - `suite_for`: Return the ordered probe specs for a target.
'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.execution.preflight.checks as checks
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema


# ----- typing aliases
ProbeFn: typing.TypeAlias = typing.Callable[
    [context.InspectionContext], schema.ProbeResult | list[schema.ProbeResult]
]


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    '''A single named check bound to an inspection-context callable.'''
    pid: str
    run: ProbeFn


# ----- helpers to build specs from check callables
def _spec(pid: str, fn, *args, **kwargs) -> ProbeSpec:
    '''Bind a check function to fixed arguments, deferring `ctx`.'''
    return ProbeSpec(pid=pid, run=lambda ctx: fn(ctx, *args, **kwargs))


def _dir(pid: str, path_attr: str) -> ProbeSpec:
    '''Build a filesystem writability spec for a resolved paths root.'''
    def _run(ctx: context.InspectionContext) -> schema.ProbeResult:
        root = _resolve_path_attr(ctx, path_attr)
        return checks.filesystem.dir_writable(root, pid)
    return ProbeSpec(pid=pid, run=_run)


def _lineage_dir(path_attr: str, pid: str) -> ProbeSpec:
    '''Build a run-history lineage spec from a paths attribute name.'''
    def _run(ctx: context.InspectionContext) -> schema.ProbeResult:
        container = _resolve_path_attr(ctx, path_attr)
        return checks.lineage.run_history(container.runs_manifest, pid)
    return ProbeSpec(pid=pid, run=_run)


def _upstream_report(pid: str, report_attr: str, target: str) -> ProbeSpec:
    '''Build an report lineage probe from a report path attribute.'''
    def run(ctx):
        fp = _resolve_path_attr(ctx, report_attr)
        return checks.lineage.upstream_report(fp, target)
    return _spec(pid, run)


def _upstream_manifest(pid: str, manifest_attr: str, target: str) -> ProbeSpec:
    '''Build an manifest lineage probe from a manifest path attribute.'''
    def run(ctx):
        fp = _resolve_path_attr(ctx, manifest_attr)
        return checks.lineage.upstream_manifest(fp, target)
    return _spec(pid, run)


def _resolve_path_attr(ctx: context.InspectionContext, dotted: str):
    '''Resolve a dotted attribute chain rooted at the artifact paths.'''
    obj: typing.Any = ctx.paths
    for part in dotted.split('.'):
        obj = getattr(obj, part)
    return obj


# ----- target suites
#
# NOTE: per-target ordering is intentional:
#   1. filesystem  — cheap, fail-fast output precondition
#   2. lineage     — upstream readiness (reports/manifests/artifacts)
#   3. policy      — execution rule sanity
#   4. contract    — domain handler validation (authoritative)
#   5. ledger      — informational state summary
#
# -----

SUITES: dict[str, tuple[ProbeSpec, ...]] = {

    'world-grid': (
        _dir('world_grid_output', 'world_grid.root'),
        _spec('grid_inputs', checks.contracts.grid_specs),
    ),

    'data-harmonize': (
        _dir('harmonization_output', 'data_harmonization.root'),
        _upstream_report(
            'upstream_world_grid',
            'world_grid.report',
            'world-grid',
        ),
        _spec('grid_reference', checks.contracts.grid_artifact),
        _spec('manifest_contract', checks.contracts.manifest_contract),
        _lineage_dir('data_harmonization', 'past_harmonization_runs'),
        _spec('harmonized_dataset', checks.lineage.pending_harmonization),
    ),

    'data-ingest': (
        _dir('ingestion_output', 'data_ingestion.root'),
        _upstream_manifest(
            'upstream_data_harmonize',
            'data_harmonization.runs_manifest',
            'data-harmonize',
        ),
        _spec('collision_policy', checks.policy.collision_policy),
        _spec('pool_grid_compatibility', checks.contracts.pool_compatibility),
        _spec('raster_alignment', checks.contracts.raster_alignment),
        _spec('feature_specs', checks.contracts.feature_specs),
        _lineage_dir('data_harmonization', 'past_harmonization_runs'),
        _lineage_dir('data_ingestion', 'past_ingestion_runs'),
        _spec('pending_ingestion', checks.lineage.pending_ingestion),
        _spec('ingested_blocks_pool', checks.ledger.ingested_pool_state),
    ),

    'data-prepare': (
        _dir('preparation_output', 'data_preparation.root'),
        _spec(
            'upstream_data_ingest_artifacts',
            lambda ctx: checks.lineage.required_artifacts(
                [
                    ctx.paths.data_ingestion.data_blocks.catalog,
                    ctx.paths.data_ingestion.data_blocks.schema,
                ],
                'data-ingest',
            ),
        ),
        _upstream_manifest(
            'upstream_data_ingest',
            'data_ingestion.runs_manifest',
            'data-ingest',
        ),
        _spec('preparation_contract', checks.contracts.preparation_contract),
        _spec('partition_config', checks.contracts.partition_config),
        _spec('ingested_blocks_pool', checks.ledger.ingested_pool_state),
        _spec('prepared_blocks_state', checks.ledger.prepared_blocks_state),
    ),

    'model-train': (
        _dir('checkpoint_dir', 'session.root'),
        _upstream_report(
            'upstream_data_prepare',
            'data_preparation.report',
            'data-prepare',
        ),
        _spec('model_body', checks.model.model_body),
        _spec(
            'pending_ingestion',
            lambda ctx: checks.lineage.pending_ingestion(
                ctx, desire_pending=False
            ),
        ),
        _spec(
            'prepared_blocks_state',
            lambda ctx: checks.ledger.prepared_blocks_state(
                ctx, desire_existing=True
            ),
        ),
        ProbeSpec(pid='hardware', run=checks.hardware.hardware_info),
    ),

    'model-evaluate': (
        _dir('eval_output', 'session.root'),
        _upstream_report(
            'upstream_data_prepare',
            'data_preparation.report',
            'data-prepare',
        ),
        _spec('model_body', checks.model.model_body),
        _spec('checkpoint_ready', checks.model.checkpoint_ready),
        _spec('eval_split', checks.model.eval_split),
        ProbeSpec(pid='hardware', run=checks.hardware.hardware_info),
    ),

    'diagnose-overfit': (
        _upstream_manifest(
            'upstream_data_harmonize',
            'data_harmonization.runs_manifest',
            'data-harmonize',
        ),
        _spec('model_body', checks.model.model_body),
        ProbeSpec(pid='hardware', run=checks.hardware.hardware_info),
    ),
}

SUPPORTED_TARGETS: tuple[str, ...] = tuple(SUITES.keys())


# ----- public functions
def suite_for(target: str) -> tuple[ProbeSpec, ...]:
    '''Return the ordered probe specs for a target.'''
    return SUITES.get(target, ())
