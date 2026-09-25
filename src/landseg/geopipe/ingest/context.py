# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Execution context resolution for data ingestion.

Provides containers and loaders to resolve the canonical spatial grid
and harmonized raster inputs for the ingestion pipeline from upstream
harmonization artifacts.

Public APIs:
    - `IngestionContext`: Container holding resolved grid and rasters.
    - `build_ingestion_context`: Load ingestion context from report.
    - `discover_successful_harmonization_runs`: Load successful runs.
    - `discover_ingested_harmonization_uids`: Ingested harm UIDs.
    - `resolve_harmonization_run`: Resolve target run record.
    - `resolve_pending_ingestion_batches`: Determine batches to ingest.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils


# ----- typing aliases
HarmonizationRecords = dict[str, contracts.HarmonizationRunRecord]
HarmonizationReportCtrl = artifacts.Controller[contracts.HarmonizationReportSchema]
HarmonizationManifestCtrl = artifacts.Controller[HarmonizationRecords]
IngestionRecords = dict[str, contracts.IngestionRunRecord]
IngestionManifestCtrl = artifacts.Controller[IngestionRecords]
DatasetSchemaCtrl = artifacts.Controller[geo_core.DatasetSchema]


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class IngestionContext:
    '''Execution context holding resolved world grid and rasters.'''
    grid: geo_core.GridLayout
    grid_fpath: str
    domains: dict[str, str] | None
    features: str | None
    labels: str | None
    valid_mask_raster: str
    run_uid: str = ''
    harmonization_run_id: str = ''
    current_run_identity: str = ''
    collided_run_uid: str | None = None

    @property
    def has_data(self) -> bool:
        '''Return True if both feature and label rasters are present.'''
        return self.features is not None and self.labels is not None


# ----- public functions
def resolve_pending_ingestion_batches(
    harmonization_run_manifest: str,
    ingestion_run_manifest: str,
    *,
    target: int | str | None = None,
    rebuild: bool = False,
) -> list[contracts.HarmonizationRunRecord]:
    '''
    Determine harmonization batches to ingest against runs manifests.

    Args:
        harmonization_run_manifest:
            File path to harmonization runs manifest JSON.
        ingestion_run_manifest:
            File path to ingestion runs manifest JSON.
        target:
            Optional harmonization target: run UID, folder name, index,
            'latest', or 'all'/'pending'/None.
        rebuild:
            If True, reprocess even if already recorded as SUCCESS.

    Returns:
        list[contracts.HarmonizationRunRecord]:
            List of harmonization run records scheduled for ingestion.

    Raises:
        artifacts.ArtifactError:
            If harmonization manifest is missing or targeted run is not
            found.
    '''
    # --------------------------------------------------------------
    # decision matrix: batch ingestion resolution
    #
    # target          manifest state        action
    # --------------  --------------------  ------------------------
    # specific run    not ingested          ingest target batch
    # specific run    ingested, rebuild=f   skip & exit (idempotent)
    # specific run    ingested, rebuild=t   re-ingest target batch
    # none (auto)     pending list non-empty auto-ingest in sequence
    # none (auto)     pending list empty    pool up-to-date (no-op)
    # 'all'/'pending' any                   catch-up pending batches
    # --------------------------------------------------------------
    successful_runs = _resolve_harmonization_runs(harmonization_run_manifest)
    ingested_uids = _resolve_ingestion_runs(ingestion_run_manifest)

    # catch-up / pending mode
    if target in (None, 'pending', 'all'):
        if rebuild:
            return list(successful_runs.values())
        return [
            rec for rec in successful_runs.values()
            if rec['run_uid'] not in ingested_uids
        ]

    # latest run mode
    if target == 'latest':
        latest_rec = list(successful_runs.values())[-1]
        if not rebuild and latest_rec['run_uid'] in ingested_uids:
            return []
        return [latest_rec]

    # targeted specific run
    target_rec = _resolve_target_harmonization_run(successful_runs, target)
    if not rebuild and target_rec['run_uid'] in ingested_uids:
        return []
    return [target_rec]

def build_ingestion_context(
    harmonization_artifact_paths: artifacts.HarmonizationPaths,
    harmonization_record: contracts.HarmonizationRunRecord,
    runs_manifest_fpath: str,
    dataset_schema_fpath: str,
    config: contracts.IngestionPipelineConfig,
) -> IngestionContext:
    '''
    Build data ingestion context from upstream harmonization artifacts.

    Discovers successful runs from the harmonization manifest, resolves
    the targeted run folder, parses out grid and raster references,
    and loads the canonical GridLayout.

    Args:
        harmonization_artifact_paths:
            File path manager for harmonization artifacts.
        harmonization_record:
            Resolved upstream harmonization run record.

    Returns:
        IngestionContext:
            Loaded execution context with world grid and input rasters.
    '''
    # parse targeted harmonization run
    harmonization_artifact_paths.get_run_folder(harmonization_record['run_id'])
    report_path = harmonization_artifact_paths.report
    report = HarmonizationReportCtrl.load_json_or_fail(report_path).fetch()

    # fetch world grid
    grid_fpath = report.get('grid_fpath')
    if not grid_fpath and 'world_grid' in report:
        grid_fpath = report['world_grid'].get('grid_fpath')
    world_grid = geo_core.load_grid_from_fpath(grid_fpath)

    # verify grid compatibility
    _verify_pool_grid_compatibility(dataset_schema_fpath, world_grid)

    # parse harmonized domain/feature/label sources
    finals = report['finalized_rasters']
    domains: dict[str, str] = {}
    for key, value in finals.items():
        if 'domain' in key:
            domains.update({key: value})

    features = finals.get('features')
    labels = finals.get('labels')

    # identify/collision check
    grid_identity = geo_utils.compute_fingerprint(world_grid.block_identity)
    identity = {
        'harmonization_run_uid': harmonization_record['run_uid'],
        'grid': {
            'grid_fpath': grid_fpath,
            'grid_identity': grid_identity,
        },
        'inputs': {
            'features': features,
            'labels': labels,
            'domains': domains,
            'valid_mask_raster': harmonization_artifact_paths.valid_mask_raster,
        },
        'config': {
            'datablocks': {
                'ignore_index': config.datablocks.ignore_index,
                'image_dem_pad': config.datablocks.image_dem_pad,
                'add_spectral': config.datablocks.add_spectral,
                'add_topo': config.datablocks.add_topo,
            },
            'domains': {
                'valid_threshold': config.domains.valid_threshold,
                'target_variance': config.domains.target_variance,
            },
        },
    }
    fingerprint = geo_utils.compute_fingerprint(identity)
    collided = geo_utils.find_run_collision(fingerprint, runs_manifest_fpath)

    return IngestionContext(
        grid=world_grid,
        grid_fpath=grid_fpath,
        domains=domains,
        features=features,
        labels=labels,
        valid_mask_raster=harmonization_artifact_paths.valid_mask_raster,
        run_uid=harmonization_record['run_uid'],
        harmonization_run_id=harmonization_record['run_id'],
        current_run_identity=fingerprint,
        collided_run_uid=collided
    )


# ----- private helpers
def _resolve_harmonization_runs(manifest_fpath: str) -> HarmonizationRecords:
    '''
    Load the harmonization runs manifest and return all successful runs.

    Args:
        manifest_fpath:
            File path to harmonization runs manifest JSON.

    Returns:
        dict[str, contracts.HarmonizationRunRecord]:
            Mapping of run UID to successful run records.
            Mapping of run UID to successful run records.

    Raises:
        artifacts.ArtifactError:
            If the manifest does not exist or has no successful runs.
    '''
    if not os.path.exists(manifest_fpath):
        raise artifacts.ArtifactError(
            'Harmonization runs manifest does not exist at '
            f'{manifest_fpath}. '
            'Please execute "data-harmonize" successfully first.'
        )

    try:
        manifest_data = HarmonizationManifestCtrl.load_json_or_fail(
            manifest_fpath
        ).fetch()
    except artifacts.ArtifactError as exc:
        raise artifacts.ArtifactError(
            'Failed to read harmonization runs manifest at '
            f'{manifest_fpath}: {exc}'
        ) from exc

    successful_runs = {
        uid: typing.cast(contracts.HarmonizationRunRecord, rec)
        for uid, rec in manifest_data.items()
        if isinstance(rec, dict) and rec.get('status') == 'SUCCESS'
    }

    if not successful_runs:
        raise artifacts.ArtifactError(
            'No successful harmonization runs found in '
            f'{manifest_fpath}. '
            'Please execute "data-harmonize" successfully first.'
        )

    return successful_runs


def _resolve_target_harmonization_run(
    successful_runs: HarmonizationRecords,
    target: int | str | None = None
) -> contracts.HarmonizationRunRecord:
    '''
    Resolve the target harmonization run record from successful runs.

    Args:
        successful_runs:
            Mapping of run UID to successful run records.
        target:
            Optional run UID, run folder name (e.g. 'run_0001'), integer
            index (e.g. 1), or directory path. If None, resolves the
            latest successful run.

    Returns:
        contracts.HarmonizationRunRecord:
            Matched successful run record.

    Raises:
        artifacts.ArtifactError:
            If the targeted run cannot be found among successful runs.
    '''
    if target is None:
        return list(successful_runs.values())[-1] # most recent run

    if isinstance(target, int) or target.isdigit():
        target_name = f'run_{int(target):04d}'
        for rec in successful_runs.values():
            if rec.get('run_id') == target_name:
                return rec
        raise artifacts.ArtifactError(
            f'Harmonization run index [{target}] ({target_name}) not '
            'found among successful runs.'
        )

    if isinstance(target, str):
        if target in successful_runs:
            return successful_runs[target]

        for rec in successful_runs.values():
            if rec.get('run_id') == target:
                return rec

        target_norm = os.path.normpath(target)
        for rec in successful_runs.values():
            rec_folder = rec.get('run_folder', '')
            if rec_folder and os.path.normpath(rec_folder) == target_norm:
                return rec

        raise artifacts.ArtifactError(
            f'Harmonization run identifier "{target}" not found among '
            'successful runs.'
        )

    raise TypeError(f'Invalid target run type: {type(target)}')


def _resolve_ingestion_runs(manifest_fpath: str) -> set[str]:
    '''
    Load ingestion runs manifest and return ingested harmonization UIDs.

    Args:
        manifest_fpath:
            File path to ingestion runs manifest JSON.

    Returns:
        set[str]:
            Set of upstream harmonization run UIDs that completed with
            status 'SUCCESS'.
    '''
    if not os.path.exists(manifest_fpath):
        return set()

    try:
        manifest_data = IngestionManifestCtrl.load_json_or_fail(
            manifest_fpath
        ).fetch()
    except artifacts.ArtifactError:
        return set()

    if not isinstance(manifest_data, dict):
        return set()

    return {
        rec['harmonization_run_uid']
        for rec in manifest_data.values()
        if isinstance(rec, dict)
        and rec.get('status') == 'SUCCESS'
        and rec.get('harmonization_run_uid')
    }


def _verify_pool_grid_compatibility(
    schema_fpath: str,
    grid: geo_core.GridLayout,
) -> None:
    '''
    Verify incoming world grid compatibility with existing block pool.

    Args:
        schema_fpath:
            File path to existing dataset schema artifact.
        grid:
            Incoming GridLayout instance to validate.

    Raises:
        artifacts.ArtifactError:
            If existing pool schema belongs to an incompatible spatial
            block identity or grid ID.
    '''
    if not os.path.exists(schema_fpath):
        return

    try:
        schema_dict = DatasetSchemaCtrl.load_json_or_fail(schema_fpath).fetch()
    except artifacts.ArtifactError as exc:
        raise artifacts.ArtifactError(
            f'Failed to read dataset schema at {schema_fpath}: {exc}'
        ) from exc

    if not isinstance(schema_dict, dict):
        return

    dataset_info = schema_dict.get('dataset', {})
    existing_block_id = dataset_info.get('block_identity')
    if existing_block_id:
        if existing_block_id != grid.block_identity:
            raise artifacts.ArtifactError(
                f'Grid compatibility mismatch: incoming batch grid '
                f'[{grid.block_identity}] does not match existing block '
                f'pool [{existing_block_id}].'
            )
        return
