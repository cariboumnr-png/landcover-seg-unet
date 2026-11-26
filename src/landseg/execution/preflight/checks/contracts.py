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
Stage contract diagnostic checks backed by `landseg.geopipe` validators.

Each check delegates domain validation through the public top-level
`landseg.geopipe` facade and adapts the plain metadata or error outcome
into a `ProbeResult`. Execution pre-flight never reconstructs internal
`geopipe` contexts or stage schemas directly.

Public APIs:
    - `feature_specs`: Validate requested feature engineering specs.
    - `grid_artifact`: Validate a persisted world grid reference.
    - `grid_specs`: Validate world grid definition inputs.
    - `manifest_contract`: Validate the input dataset manifest.
    - `pool_compatibility`: Validate pool grid compatibility.
    - `prepare_semantics`: Validate feature and target resolution.
    - `raster_alignment`: Validate raster/grid alignment for ingestion.
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema
import landseg.geopipe as geopipe


# ----- constants
_CATEGORY = schema.ProbeCategory.CONTRACT


# ----- public functions
def grid_specs(
    ctx: context.InspectionContext,
    pid: str = 'grid_inputs',
) -> schema.ProbeResult:
    '''
    Validate world grid definition inputs via the domain handler.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome.
    '''
    try:
        details = geopipe.validate_grid_specs(ctx.config.data.world_grid)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(pid, f'World grid inputs invalid: {err}', err)

    return _pass(pid, 'World grid inputs verified', **details)


def grid_artifact(
    ctx: context.InspectionContext,
    pid: str = 'grid_reference',
) -> schema.ProbeResult:
    '''
    Validate a persisted world grid reference artifact.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome.
    '''
    report_fp = ctx.paths.world_grid.report
    try:
        info = geopipe.validate_grid_reference(report_fp)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'World grid reference validation failed: {err}',
            err,
            report_fpath=str(report_fp),
        )

    return _pass(
        pid,
        (
            f'World grid reference verified (grid_id: {info["grid_id"]}, '
            f'tiles: {info["tiles"]})'
        ),
        grid_id=info['grid_id'],
        grid_fpath=info['grid_fpath'],
        report_fpath=info['report_fpath'],
    )


def manifest_contract(
    ctx: context.InspectionContext,
    pid: str = 'manifest_contract',
) -> schema.ProbeResult:
    '''
    Validate the configured input dataset manifest contract.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome.
    '''
    manifest_fp = ctx.config.data.harmonization.dataset_manifest
    if not manifest_fp:
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message='Dataset manifest path is not configured',
        )

    try:
        info = geopipe.validate_manifest_contract(pathlib.Path(manifest_fp))
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'Dataset manifest contract invalid: {err}',
            err,
            dataset_manifest=str(manifest_fp),
        )

    return _pass(
        pid,
        (
            f'Dataset manifest contract verified '
            f'({info["entry_count"]} valid entries)'
        ),
        dataset_manifest=info['dataset_manifest'],
    )


def feature_specs(
    ctx: context.InspectionContext,
    pid: str = 'feature_specs',
) -> schema.ProbeResult:
    '''
    Validate requested feature engineering specs against bands.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome, or SKIP if unneeded.
    '''
    ing_cfg = ctx.config.data.ingestion
    datablocks = ing_cfg.datablocks
    try:
        res = geopipe.validate_feature_specs(
            ctx.paths.data_harmonization,
            ctx.paths.data_ingestion,
            target=ing_cfg.harmonization_run,
            rebuild=ing_cfg.rebuild,
            add_spectral=datablocks.add_spectral,
            add_topo=datablocks.add_topo,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(pid, f'Feature engineering specs invalid: {err}', err)

    if res['status'] == 'skip':
        return _skip(pid, res['message'])

    return _pass(
        pid,
        'Feature engineering specs are feasible',
        add_spectral=res['add_spectral'],
        add_topo=res['add_topo'],
    )


def pool_compatibility(
    ctx: context.InspectionContext,
    pid: str = 'pool_grid_compatibility',
) -> schema.ProbeResult:
    '''
    Validate world grid compatibility with the ingested pool.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome.
    '''
    schema_fp = ctx.paths.data_ingestion.data_blocks.schema
    try:
        report_fp: pathlib.Path | None = ctx.paths.world_grid.report
    except ValueError:
        # world_grid root unconfigured in decoupled data-ingest
        report_fp = None

    if report_fp is None or not report_fp.exists():
        return _pass(
            pid,
            'Upstream world grid report decoupled; pool check deferred '
            'to ingestion runtime',
            report_fpath=str(report_fp) if report_fp else None,
        )

    try:
        info = geopipe.validate_pool_compatibility(schema_fp, report_fp)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'Pool grid compatibility invalid: {err}',
            err,
            schema_fpath=str(schema_fp),
            report_fpath=str(report_fp),
        )

    message = (
        f'Pool grid compatibility verified (grid_id: {info["grid_id"]})'
        if info['schema_exists']
        else 'Pool schema not yet initialized (baseline established)'
    )
    return _pass(pid, message, **info)


def raster_alignment(
    ctx: context.InspectionContext,
    pid: str = 'raster_alignment',
) -> schema.ProbeResult:
    '''
    Validate that harmonized rasters align with their world grid.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome, or SKIP if unneeded.
    '''
    ing_cfg = ctx.config.data.ingestion
    try:
        res = geopipe.validate_raster_alignment(
            ctx.paths.data_harmonization,
            ctx.paths.data_ingestion,
            target=ing_cfg.harmonization_run,
            rebuild=ing_cfg.rebuild,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(pid, f'Raster/grid alignment invalid: {err}', err)

    if res['status'] == 'skip':
        return _skip(pid, res['message'])

    return _pass(
        pid,
        'Harmonized rasters align with the world grid',
        features=res['features'],
        labels=res['labels'],
        grid_id=res['grid_id'],
    )


def prepare_semantics(
    ctx: context.InspectionContext,
    pid: str = 'preparation_contract',
) -> schema.ProbeResult:
    '''
    Validate feature channel and target head resolution.

    Args:
        ctx:
            Pre-resolved inspection context.
        pid:
            Probe identifier for the returned result.

    Returns:
        schema.ProbeResult:
            Contract validation probe outcome.
    '''
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
        info = geopipe.validate_prepare_semantics(
            schema_fp,
            ctx.config.data.preparation.datasetview,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return _fail(
            pid,
            f'Preparation contract invalid: {err}',
            err,
            schema_fpath=str(schema_fp),
        )

    return _pass(
        pid,
        (
            f'Preparation contract verified (focal_head: {info["focal_head"]}, '
            f'features: {len(info["features"])}, '
            f'target_heads: {len(info["target_heads"])})'
        ),
        **info,
    )


# ----- private helpers
def _pass(pid: str, message: str, **details: typing.Any) -> schema.ProbeResult:
    '''Build a standard CONTRACT passing probe result.'''
    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=message,
        details=details,
    )


def _skip(pid: str, message: str) -> schema.ProbeResult:
    '''Build a standard CONTRACT skipped probe result.'''
    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.SKIP,
        message=message,
    )


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
