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
Pre-flight and stage-contract validation facade for `landseg.geopipe`.

Provides lightweight, plain-data validation entrypoints over internal
stage validators (`grid`, `harmonize`, `ingest`, `prepare`) so external
callers never assemble `IngestionContext` or inspect internal schema
payloads directly.

Public APIs:
    - `validate_feature_specs`: Validate requested feature specs.
    - `validate_grid_reference`: Validate persisted grid reference.
    - `validate_grid_specs`: Validate world grid definition inputs.
    - `validate_manifest_contract`: Validate dataset manifest entries.
    - `validate_pool_compatibility`: Validate grid/pool compatibility.
    - `validate_prepare_semantics`: Validate prepare feature/head specs.
    - `validate_raster_alignment`: Validate raster and grid alignment.
'''

# standard imports
from __future__ import annotations
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid as grid_mod
import landseg.geopipe.harmonize as harmonize_mod
import landseg.geopipe.ingest as ingest_mod
import landseg.geopipe.prepare as prepare_mod


# ----- public functions
def validate_grid_specs(
    config: contracts.WorldGridPrepConfig,
) -> dict[str, typing.Any]:
    '''
    Validate world grid generation inputs and return detail mapping.

    Args:
        config:
            World grid preparation configuration object.

    Returns:
        dict[str, typing.Any]:
            Dictionary of validated world grid specification fields.
    '''
    specs = grid_mod.validators.grid_specs(config)
    return _as_details(specs)


def validate_grid_reference(
    report_fpath: pathlib.Path,
) -> dict[str, typing.Any]:
    '''
    Validate a persisted world grid reference and return summary.

    Args:
        report_fpath:
            Path to the persisted world grid JSON report.

    Returns:
        dict[str, typing.Any]:
            Summary metadata with grid ID, tile count, and paths.
    '''
    grid, gid, grid_fp = harmonize_mod.validators.grid_reference(report_fpath)
    return {
        'grid_id': gid,
        'tiles': len(grid),
        'grid_fpath': str(grid_fp),
        'report_fpath': str(report_fpath),
    }


def validate_manifest_contract(
    manifest_fpath: pathlib.Path,
) -> dict[str, typing.Any]:
    '''
    Validate the input dataset manifest contract and return counts.

    Args:
        manifest_fpath:
            Path to the input dataset manifest JSON or YAML file.

    Returns:
        dict[str, typing.Any]:
            Summary mapping containing validated entry count and path.
    '''
    entries = harmonize_mod.validators.manifest_contract(manifest_fpath)
    return {
        'entry_count': len(entries),
        'dataset_manifest': str(manifest_fpath),
    }


def validate_pool_compatibility(
    schema_fpath: pathlib.Path,
    report_fpath: pathlib.Path,
) -> dict[str, typing.Any]:
    '''
    Validate world grid compatibility with an existing ingestion pool.

    Args:
        schema_fpath:
            Path to the existing ingestion `schema.json` file.
        report_fpath:
            Path to the target world grid JSON report.

    Returns:
        dict[str, typing.Any]:
            Summary metadata with grid ID, schema path, and existence.

    Raises:
        ValueError:
            If loading or validating the world grid reference fails.
    '''
    try:
        target_grid = harmonize_mod.validators.grid_reference(report_fpath)[0]
    except Exception as err:
        raise ValueError(f'World grid reference invalid: {err}') from err

    ingest_mod.validators.pool_compatibility(schema_fpath, target_grid)
    return {
        'grid_id': target_grid.gid,
        'schema_fpath': str(schema_fpath),
        'schema_exists': schema_fpath.exists(),
    }


def validate_feature_specs(  # pylint: disable=too-many-arguments
    harmonization_paths: artifacts.HarmonizationPaths,
    ingestion_paths: artifacts.IngestionPaths,
    *,
    target: int | str | None = 'latest',
    rebuild: bool = False,
    add_spectral: list[str] | None = None,
    add_topo: list[str] | None = None,
) -> dict[str, typing.Any]:
    '''
    Validate requested feature engineering specs against batch bands.

    Returns a plain outcome dictionary with `'status'` equal to
    `'skip'` (plus `'message'`) when no pending harmonization batch
    can be resolved, or `'pass'` with the validated spec fields when
    feasible.

    Args:
        harmonization_paths:
            Path manager for upstream harmonization artifacts.
        ingestion_paths:
            Path manager for ingestion runs manifest and outputs.
        target:
            Target harmonization run identifier or index.
        rebuild:
            Whether to consider already-ingested batches for rebuild.
        add_spectral:
            Optional spectral index names requested for ingestion.
        add_topo:
            Optional topographic derivative names requested for ingest.

    Returns:
        dict[str, typing.Any]:
            Plain outcome summary indicating `'skip'` or `'pass'`.
    '''
    ingest_ctx, skip_message = _resolve_validation_ingestion_context(
        harmonization_paths,
        ingestion_paths,
        target=target,
        rebuild=rebuild,
    )
    if ingest_ctx is None:
        return {'status': 'skip', 'message': skip_message}

    ingest_mod.validators.feature_specs(
        ingest_ctx.features,  # type: ignore[arg-type]
        ingest_ctx.labels,
        add_spectral=add_spectral,
        add_topo=add_topo,
    )
    return {
        'status': 'pass',
        'add_spectral': add_spectral,
        'add_topo': add_topo,
    }


def validate_raster_alignment(
    harmonization_paths: artifacts.HarmonizationPaths,
    ingestion_paths: artifacts.IngestionPaths,
    *,
    target: int | str | None = 'latest',
    rebuild: bool = False,
) -> dict[str, typing.Any]:
    '''
    Validate harmonized rasters align with their world grid for ingest.

    Returns a plain outcome dictionary with `'status'` equal to
    `'skip'` (plus `'message'`) when no pending harmonization batch
    can be resolved, or `'pass'` with aligned raster/grid metadata.

    Args:
        harmonization_paths:
            Path manager for upstream harmonization artifacts.
        ingestion_paths:
            Path manager for ingestion runs manifest and outputs.
        target:
            Target harmonization run identifier or index.
        rebuild:
            Whether to consider already-ingested batches for rebuild.

    Returns:
        dict[str, typing.Any]:
            Plain outcome summary indicating `'skip'` or `'pass'`.
    '''
    ingest_ctx, skip_message = _resolve_validation_ingestion_context(
        harmonization_paths,
        ingestion_paths,
        target=target,
        rebuild=rebuild,
    )
    if ingest_ctx is None:
        return {'status': 'skip', 'message': skip_message}

    ingest_mod.validators.raster_alignment(
        ingest_ctx.features,  # type: ignore[arg-type]
        ingest_ctx.grid,
        ingest_ctx.labels,
    )
    return {
        'status': 'pass',
        'features': str(ingest_ctx.features),
        'labels': str(ingest_ctx.labels) if ingest_ctx.labels else None,
        'grid_id': ingest_ctx.grid.gid,
    }


def validate_prepare_semantics(
    schema_fpath: pathlib.Path,
    config: contracts.PreparationDatasetViewConfig | None = None,
) -> dict[str, typing.Any]:
    '''
    Load ingestion data schema and validate preparation semantics.

    Args:
        schema_fpath:
            Path to the ingestion `schema.json` file.
        config:
            Optional preparation dataset view configuration object.

    Returns:
        dict[str, typing.Any]:
            Validated focal head, feature names, target heads, and path.
    '''
    Ctrl = artifacts.Controller[geo_core.DataSchema]
    schm = Ctrl(schema_fpath, schema_id=geo_core.data_schema.SCHEMA_ID).fetch()
    fts, targets, focal = prepare_mod.validators.dataset_semantics(schm, config)
    return {
        'focal_head': focal,
        'features': list(fts.names),
        'target_heads': list(targets.head_names),
        'schema_fpath': str(schema_fpath),
    }


# ----- private helpers
def _resolve_validation_ingestion_context(
    harmonization_paths: artifacts.HarmonizationPaths,
    ingestion_paths: artifacts.IngestionPaths,
    *,
    target: int | str | None = 'latest',
    rebuild: bool = False,
) -> tuple[ingest_mod.IngestionContext | None, str | None]:
    '''Resolve pending batch record and build ingestion context.'''
    resolved_target = target or 'latest'
    try:
        batches = ingest_mod.resolve_pending_ingestion_batches(
            harmonization_paths.runs_manifest,
            ingestion_paths.runs_manifest,
            target=resolved_target,
            rebuild=rebuild,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return None, f'Skipped: unable to resolve harmonization batch ({err})'

    if not batches:
        return None, (
            'Skipped: unable to resolve harmonization batch '
            '(No pending harmonization runs found for target '
            f'"{resolved_target}".)'
        )

    ingest_ctx = ingest_mod.build_ingestion_context(
        harmonization_paths,
        batches[0],
    )
    if ingest_ctx.features is None:
        raise ValueError('No harmonized feature raster resolved for ingestion')
    return ingest_ctx, None


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
