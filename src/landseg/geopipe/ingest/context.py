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
'''

# standard imports
from __future__ import annotations
import dataclasses
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.harmonization as contracts
import landseg.geopipe.core as geo_core


# ----- typing aliases
ReportCtrl = artifacts.Controller[contracts.HarmonizationReportSchema]


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

    @property
    def has_data(self) -> bool:
        '''Return True if both feature and label rasters are present.'''
        return self.features is not None and self.labels is not None


# ----- public functions
def build_ingestion_context(
    harmonization_paths: artifacts.HarmonizationPaths,
    harmonization_run_id: int | str | None = None
) -> IngestionContext:
    '''
    Build data ingestion context from upstream harmonization artifacts.

    Locates the targeted harmonization run folder, parses out the grid
    and raster artifact references from the report, and loads the
    canonical GridLayout layout.

    Args:
        harmonization_paths:
            File path manager for harmonization artifacts.
        harmonization_run_id:
            Target run identifier, or None to use the latest run.

    Returns:
        IngestionContext:
            Loaded execution context with world grid and input rasters.
    '''
    # locate targeted/latest harmonization run folder
    harmonization_paths.get_run_folder(harmonization_run_id)

    # read report into a typed dict
    report_path = harmonization_paths.report
    report = ReportCtrl.load_json_or_fail(report_path).fetch()

    finals = report['finalized_rasters']
    assert finals

    grid_fpath = report.get('grid_fpath')
    if not grid_fpath and 'world_grid' in report:
        grid_fpath = report['world_grid'].get('grid_fpath')
    assert grid_fpath

    # load canonical world grid using core loader
    world_grid = geo_core.load_grid_from_fpath(grid_fpath)

    # see if domains are present
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
    )
