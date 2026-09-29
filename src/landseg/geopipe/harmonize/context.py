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
Execution context resolution for raster harmonization.

Provides containers and loaders to resolve the canonical spatial grid
reference for data harmonization from upstream pipeline artifacts.

Public APIs:
    - `HarmonizationContext`: Container holding resolved world grid.
    - `build_harmonization_context`: Load grid context from report.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.utils as geo_utils


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class HarmonizationContext:
    '''Resolved world grid reference container for harmonization.'''
    grid: geo_core.GridLayout
    grid_id: str
    grid_fpath: str
    compiled_dataset_manifest: dict[str, manifest.ManifestEntry]
    current_run_identity: str = ''
    collided_run_uid: str | None = None

# ----- public functions
def build_harmonization_context(
    grid_source_path: str,
    runs_manifest_fpath: str,
    config: contracts.HarmonizationPipelineConfig,
) -> HarmonizationContext:
    '''
    Load world grid reference context from an upstream grid report.

    Args:
        grid_source:
            File path to grid report JSON or directory containing it.

    Returns:
        HarmonizationContext:
            Execution context containing the restored GridLayout and
            grid reference metadata.
    '''
    # fetch world grid
    if os.path.isdir(grid_source_path):
        report_fpath = geo_core.get_grid_report_fpath(grid_source_path)
    else:
        report_fpath = grid_source_path
    grid_report = geo_core.read_grid_report(report_fpath)
    world_grid = geo_core.load_grid_from_fpath(grid_report['grid_fpath'])

    # compile dataset manifest JOSN
    compiled = manifest.compile_dataset_manifest(config.dataset_manifest)

    # identify/collision check
    grid_identity = geo_utils.compute_fingerprint(world_grid.affine_identity)
    identity = {
        'inputs': compiled,
        'grid': {
            'grid_fpath': report_fpath,
            'grid_identity': grid_identity,
        },
        'config': {
            'categorical_resampling': config.resampling_categorical,
            'continuous_resampling': config.resampling_continuous,
        }
    }
    fingerprint = geo_utils.compute_fingerprint(identity)
    collided = geo_utils.find_run_collision(fingerprint, runs_manifest_fpath)

    return HarmonizationContext(
        grid=world_grid,
        grid_id=grid_report['grid_id'],
        grid_fpath=grid_report['grid_fpath'],
        compiled_dataset_manifest=compiled,
        current_run_identity=fingerprint,
        collided_run_uid=collided
    )
