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
Adapter bridging harmonization outputs into the ingestion pipeline.

Parses finalized raster paths and world grid definitions from the
harmonization report, providing typed inputs to subsequent domain mapping
and block construction stages.

Public APIs:
    - HarmonizedRasters: Dataclass container for harmonized outputs.
    - read_harmonization_report: Reads report to extract rasters.
'''

# standard imports
import dataclasses
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize as harmonize


# ----- typing aliases
ReportController = artifacts.Controller[harmonize.HarmonizationReportSchema]


# ----- public dataclasses
@dataclasses.dataclass
class HarmonizedRasters:
    '''Container for harmonized rasters read from the report.'''
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
def read_harmonization_report(
    harmonization_paths: artifacts.HarmonizationPaths,
    harmonization_run_id: int | str | None
) -> HarmonizedRasters:
    '''
    Read harmonization report to extract finalized rasters.

    Locates the targeted harmonization run folder, fetches the
    corresponding report artifact, and parses out world grid, domain,
    feature, and label raster references.

    Args:
        harmonization_paths:
            File path manager for harmonization artifacts.
        harmonization_run_id:
            Target run identifier, or None to use the latest run.

    Returns:
        HarmonizedRasters:
            Parsed container holding paths to finalized rasters.
    '''
    # locate targeted/latest harmonization run folder
    harmonization_paths.get_run_folder(harmonization_run_id)

    # read report into a typed dict
    report_path = harmonization_paths.report
    report = ReportController.load_json_or_fail(report_path).fetch()

    finals = report['finalized_rasters']
    assert finals

    world_grid = report.get('world_grid')
    assert world_grid

    # see if domains are present
    domains: dict[str, str] = {}
    for key, value in finals.items():
        if 'domain' in key: # search by tag
            domains.update({key: value})

    features = finals.get('features')
    labels = finals.get('labels')

    return HarmonizedRasters(
        grid_fpath=world_grid['grid_fpath'],
        domains=domains,
        features=features,
        labels=labels,
        valid_mask_raster=harmonization_paths.valid_mask_raster,
    )
