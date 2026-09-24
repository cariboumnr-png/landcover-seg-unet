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

# pylint: disable=missing-function-docstring

'''
Canonical filesystem paths for data harmonization (ETL) artifacts.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass
class HarmonizationPaths(base.PipelineArtifactsPaths):
    '''Paths for data harmonization ETL artifacts.'''

    @property
    def runs_manifest(self) -> str:
        return os.path.join(self.root, 'harmonization_runs.json')

    @property
    def valid_mask_raster(self) -> str:
        return os.path.join(self.effective_run_folder, 'valid_pixel_mask.vrt')

    @property
    def report(self) -> str:
        return os.path.join(self.effective_run_folder, 'harmonize_report.json')

    @property
    def config(self) -> str:
        return os.path.join(self.effective_run_folder, 'config.json')

    def _init_pipeline_folders(self):
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.effective_run_folder, exist_ok=True)
        if not os.path.exists(self.runs_manifest):
            controller.Controller[dict](self.runs_manifest).persist({})
