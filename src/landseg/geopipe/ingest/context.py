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
    - `resolve_harmonization_run`: Resolve target run record.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.harmonization as contracts
import landseg.geopipe.core as geo_core


# ----- typing aliases
ReportCtrl = artifacts.Controller[contracts.HarmonizationReportSchema]
ManifestCtrl = artifacts.Controller[dict[str, typing.Any]]


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

    @property
    def has_data(self) -> bool:
        '''Return True if both feature and label rasters are present.'''
        return self.features is not None and self.labels is not None


# ----- public functions
def discover_successful_harmonization_runs(
    harmonization_paths: artifacts.HarmonizationPaths,
) -> dict[str, contracts.HarmonizationRunRecord]:
    '''
    Load the harmonization runs manifest and return all successful runs.

    Args:
        harmonization_paths:
            File path manager for harmonization artifacts.

    Returns:
        dict[str, contracts.HarmonizationRunRecord]:
            Mapping of run UID to successful run records.

    Raises:
        artifacts.ArtifactError:
            If the manifest does not exist or has no successful runs.
    '''
    manifest_fpath = harmonization_paths.runs_manifest
    if not os.path.exists(manifest_fpath):
        raise artifacts.ArtifactError(
            'Harmonization runs manifest does not exist at '
            f'{manifest_fpath}. '
            'Please execute "data-harmonize" successfully first.'
        )

    try:
        manifest_data = ManifestCtrl.load_json_or_fail(
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


def resolve_harmonization_run(
    successful_runs: dict[str, contracts.HarmonizationRunRecord],
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
        return list(successful_runs.values())[-1]

    if isinstance(target, int):
        target_name = f'run_{target:04d}'
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

        if target.isdigit():
            target_name = f'run_{int(target):04d}'
            for rec in successful_runs.values():
                if rec.get('run_id') == target_name:
                    return rec

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


def build_ingestion_context(
    harmonization_paths: artifacts.HarmonizationPaths,
    harmonization_run_id: int | str | None = None
) -> IngestionContext:
    '''
    Build data ingestion context from upstream harmonization artifacts.

    Discovers successful runs from the harmonization manifest, resolves
    the targeted run folder, parses out grid and raster references,
    and loads the canonical GridLayout.

    Args:
        harmonization_paths:
            File path manager for harmonization artifacts.
        harmonization_run_id:
            Target run identifier, or None to use the latest run.

    Returns:
        IngestionContext:
            Loaded execution context with world grid and input rasters.
    '''
    successful_runs = discover_successful_harmonization_runs(
        harmonization_paths
    )
    target_record = resolve_harmonization_run(
        successful_runs, harmonization_run_id
    )

    harmonization_paths.get_run_folder(target_record['run_folder'])

    report_path = harmonization_paths.report
    report = ReportCtrl.load_json_or_fail(report_path).fetch()

    finals = report['finalized_rasters']
    assert finals

    grid_fpath = report.get('grid_fpath')
    if not grid_fpath and 'world_grid' in report:
        grid_fpath = report['world_grid'].get('grid_fpath')
    assert grid_fpath

    world_grid = geo_core.load_grid_from_fpath(grid_fpath)

    domains: dict[str, str] = {}
    for key, value in finals.items():
        if 'domain' in key:
            domains.update({key: value})

    features = finals.get('features')
    labels = finals.get('labels')

    return IngestionContext(
        grid=world_grid,
        grid_fpath=grid_fpath,
        domains=domains,
        features=features,
        labels=labels,
        valid_mask_raster=harmonization_paths.valid_mask_raster,
        run_uid=target_record['run_uid'],
        harmonization_run_id=target_record['run_id'],
    )
