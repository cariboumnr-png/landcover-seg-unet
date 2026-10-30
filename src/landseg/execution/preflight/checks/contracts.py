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
Stage contract diagnostic checks backed by authoritative domain handlers.

Each check is a thin adapter: it reconstructs the same inputs the
target pipeline assembles at runtime, delegates the actual validation
to the stage package's `validators` namespace, and converts the outcome
into a `ProbeResult`. This guarantees preflight and execution share a
single source of domain truth.

Public APIs:
    - `grid_specs`: Validate world grid definition inputs.
    - `grid_artifact`: Validate a persisted world grid reference.
    - `manifest_contract`: Validate the input dataset manifest.
    - `pool_compatibility`: Validate pool grid compatibility.
    - `raster_alignment`: Validate raster/grid alignment for ingestion.
    - `feature_specs`: Validate requested feature engineering specs.
    - `preparation_contract`: Validate feature/target resolution.
    - `partition_config`: Validate partition ratios, stride, and AOIs.
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid as grid_mod
import landseg.geopipe.harmonize as harmonize_mod
import landseg.geopipe.ingest as ingest_mod
import landseg.geopipe.prepare as prepare_mod


# ----- constants
_CATEGORY = schema.ProbeCategory.CONTRACT


# ----- public functions
# geopipe.grid
def grid_specs(
    ctx: context.InspectionContext,
    pid: str = 'grid_inputs',
) -> schema.ProbeResult:
    '''Validate world grid definition inputs through the domain handler.'''
    try:
        specs = grid_mod.validators.grid_reference(ctx.config.data.world_grid)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(pid, f'World grid inputs invalid: {err}', err)

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message='World grid inputs verified',
        details=_as_details(specs),
    )


def grid_artifact(
    ctx: context.InspectionContext,
    pid: str = 'grid_reference',
) -> schema.ProbeResult:
    '''Validate a persisted world grid reference artifact.'''
    report_fp = ctx.paths.world_grid.report
    try:
        grid, gid, grid_fp = harmonize_mod.validators.grid_reference(report_fp)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'World grid reference validation failed: {err}',
            err,
            report_fpath=str(report_fp),
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=(
            f'World grid reference verified (grid_id: {gid}, '
            f'tiles: {len(grid)})'
        ),
        details={
            'grid_id': gid,
            'grid_fpath': str(grid_fp),
            'report_fpath': str(report_fp),
        },
    )


# geopipe.harmonize
def manifest_contract(
    ctx: context.InspectionContext,
    pid: str = 'manifest_contract',
) -> schema.ProbeResult:
    '''Validate the configured input dataset manifest contract.'''
    manifest_fp = ctx.config.data.harmonization.dataset_manifest
    if not manifest_fp:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message='Dataset manifest path is not configured',
        )

    try:
        entries = harmonize_mod.validators.manifest_contract(
            pathlib.Path(manifest_fp)
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'Dataset manifest contract invalid: {err}',
            err,
            dataset_manifest=str(manifest_fp),
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=(
            f'Dataset manifest contract verified '
            f'({len(entries)} valid entries)'
        ),
        details={'dataset_manifest': str(manifest_fp)},
    )


# geopipe.ingest
def pool_compatibility(
    ctx: context.InspectionContext,
    pid: str = 'pool_grid_compatibility',
) -> schema.ProbeResult:
    '''Validate that the world grid is compatible with the ingested pool.'''
    schema_fp = ctx.paths.data_ingestion.data_blocks.schema
    report_fp = ctx.paths.world_grid.report

    if not report_fp.exists():
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=f'World grid report not found at {report_fp}',
            details={'report_fpath': str(report_fp)},
        )

    try:
        target_grid = harmonize_mod.validators.grid_reference(report_fp)[0]
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'World grid reference invalid: {err}',
            err,
            report_fpath=str(report_fp),
        )

    try:
        ingest_mod.validators.pool_compatibility(schema_fp, target_grid)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'Pool grid compatibility invalid: {err}',
            err,
            schema_fpath=str(schema_fp),
        )

    message = (
        'Pool schema not yet initialized (baseline established)'
        if not schema_fp.exists()
        else f'Pool grid compatibility verified (grid_id: {target_grid.gid})'
    )
    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=message,
        details={
            'grid_id': target_grid.gid,
            'schema_fpath': str(schema_fp),
            'schema_exists': schema_fp.exists(),
        },
    )


def raster_alignment(
    ctx: context.InspectionContext,
    pid: str = 'raster_alignment',
) -> schema.ProbeResult:
    '''
    Validate harmonized rasters align with their world grid.

    Resolves the target harmonization batch exactly as ingestion would,
    then delegates CRS/resolution/overlap validation to the ingest
    handler. Read-only: no collision manifest or schema reads occur.
    '''
    paths = ctx.paths
    try:
        record = _resolve_ingestion_record(ctx)
    except Exception as err:  # pylint: disable=broad-exception-caught
        # lineage checks report the missing-upstream failure separately;
        # here we simply cannot evaluate alignment yet.
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.SKIP,
            message=f'Skipped: unable to resolve harmonization batch ({err})',
        )

    try:
        ingest_ctx = ingest_mod.build_ingestion_context(
            paths.data_harmonization,
            record,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid, f'Unable to build ingestion context: {err}', err
        )

    if ingest_ctx.features is None:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message='No harmonized feature raster resolved for ingestion',
        )

    try:
        ingest_mod.validators.raster_alignment(
            ingest_ctx.features,
            ingest_ctx.grid,
            ingest_ctx.labels,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(pid, f'Raster/grid alignment invalid: {err}', err)

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message='Harmonized rasters align with the world grid',
        details={
            'features': str(ingest_ctx.features),
            'labels': (
                str(ingest_ctx.labels) if ingest_ctx.labels else None
            ),
            'grid_id': ingest_ctx.grid.gid,
        },
    )


def feature_specs(
    ctx: context.InspectionContext,
    pid: str = 'feature_specs',
) -> schema.ProbeResult:
    '''
    Validate requested feature engineering specs against actual bands.

    Confirms that spectral/topographic add-ons requested in config are
    feasible given the bands present in the harmonized rasters.
    '''
    paths = ctx.paths
    datablocks = ctx.config.data.ingestion.datablocks

    try:
        record = _resolve_ingestion_record(ctx)
        ingest_ctx = ingest_mod.build_ingestion_context(
            paths.data_harmonization,
            record,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.SKIP,
            message=f'Skipped: unable to resolve harmonization batch ({err})',
        )

    if ingest_ctx.features is None:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message='No harmonized feature raster resolved for ingestion',
        )

    try:
        ingest_mod.validators.feature_specs(
            ingest_ctx.features,
            ingest_ctx.labels,
            add_spectral=datablocks.add_spectral,
            add_topo=datablocks.add_topo,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(pid, f'Feature engineering specs invalid: {err}', err)

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message='Feature engineering specs are feasible',
        details={
            'add_spectral': datablocks.add_spectral,
            'add_topo': datablocks.add_topo,
        },
    )


# geopipe.prepare
def preparation_contract(
    ctx: context.InspectionContext,
    pid: str = 'preparation_contract',
) -> schema.ProbeResult:
    '''Validate feature channel and target head resolution.'''
    schema_fp = ctx.paths.data_ingestion.data_blocks.schema
    if not schema_fp.exists():
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=f'Upstream ingestion schema not found at {schema_fp}',
            details={'schema_fpath': str(schema_fp)},
        )

    try:
        Ctrl = artifacts.Controller[geo_core.DataSchema]
        data_schema = Ctrl(
            schema_fp, schema_id=geo_core.data_schema.SCHEMA_ID
        ).fetch()
        prep_cfg = ctx.config.data.preparation

        features = prepare_mod.validators.feature_channels(
            data_schema, prep_cfg.datasetview.features
        )
        targets = prepare_mod.validators.target_heads(
            data_schema, prep_cfg.datasetview.targets
        )
        focal_head = prepare_mod.validators.focal_head(
            targets, prep_cfg.datasetview.focal_target
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'Preparation contract invalid: {err}',
            err,
            schema_fpath=str(schema_fp),
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=(
            f'Preparation contract verified (focal_head: {focal_head}, '
            f'features: {len(features.names)}, '
            f'target_heads: {len(targets.head_names)})'
        ),
        details={
            'focal_head': focal_head,
            'features': list(features.names),
            'target_heads': list(targets.head_names),
            'schema_fpath': str(schema_fp),
        },
    )


def partition_config(
    ctx: context.InspectionContext,
    pid: str = 'partition_config',
) -> schema.ProbeResult:
    '''
    Validate partition ratios, stride, and AOI availability.

    Checks the domain invariants the partitioner relies on but the
    config schema does not enforce: val+test < 1, positive buffer
    stride, and presence of configured AOI rasters.
    '''
    partition = ctx.config.data.preparation.partition

    problems: list[str] = []

    if partition.val_ratio < 0 or partition.test_ratio < 0:
        problems.append(
            f'neg ratio (val={partition.val_ratio}, '
            f'test={partition.test_ratio})'
        )
    if partition.val_ratio + partition.test_ratio >= 1.0:
        problems.append(
            f'val+test ratios must be < 1 (got '
            f'{partition.val_ratio + partition.test_ratio:.3f})'
        )
    if partition.buffer_step < 0:
        problems.append(
            f'buffer_step must be >= 0 (got {partition.buffer_step})'
        )

    for aoi_name in ('train_aoi', 'val_aoi', 'test_aoi'):
        aoi_val = getattr(partition, aoi_name, None)
        if aoi_val and not pathlib.Path(aoi_val).exists():
            problems.append(f'{aoi_name} not found at {aoi_val}')

    if problems:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=f'Partition configuration invalid: {"; ".join(problems)}',
            details={'problems': problems},
        )

    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=(
            f'Partition config verified (val={partition.val_ratio}, '
            f'test={partition.test_ratio}, '
            f'buffer_step={partition.buffer_step})'
        ),
        details={
            'val_ratio': partition.val_ratio,
            'test_ratio': partition.test_ratio,
            'buffer_step': partition.buffer_step,
        },
    )


# ----- private helpers
def _resolve_ingestion_record(
    ctx: context.InspectionContext,
) -> contracts.HarmonizationRunRecord:
    '''Resolve the harmonization run record ingestion would target.'''
    paths = ctx.paths
    target = ctx.config.data.ingestion.harmonization_run or 'latest'
    batches = ingest_mod.resolve_pending_ingestion_batches(
        paths.data_harmonization.runs_manifest,
        paths.data_ingestion.runs_manifest,
        target=target,
        rebuild=ctx.config.data.ingestion.rebuild,
    )
    return batches[0]


def _fail(
    pid: str,
    message: str,
    err: Exception,
    **details: typing.Any,
) -> schema.ProbeResult:
    '''Build a standard CONTRACT failure probe result.'''
    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.FAIL,
        message=message,
        details={**details, 'error': str(err)},
    )


def _as_details(obj: object) -> dict[str, typing.Any]:
    '''Best-effort conversion of a specs object to a details mapping.'''
    as_dict = getattr(obj, 'as_dict', None)
    if callable(as_dict):
        try:
            result = as_dict()
        except Exception:  # pylint: disable=broad-exception-caught
            return {}
        if isinstance(result, dict):
            return dict(result)
    return {}
