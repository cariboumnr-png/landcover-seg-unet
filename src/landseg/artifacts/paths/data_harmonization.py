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

'''Canonical filesystem paths for data harmonization ETL artifacts.'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass
class HarmonizationPaths(base.PipelineArtifactsPaths):
    '''Paths for data harmonization ETL artifacts.'''
    @property
    def runs_manifest(self) -> pathlib.Path:
        '''Return path to the harmonization runs manifest JSON.'''
        return self.root / 'harmonization_runs.json'

    @property
    def valid_mask_raster(self) -> pathlib.Path:
        '''Return path to the harmonized valid pixel mask VRT.'''
        return self.effective_run_folder / 'valid_pixel_mask.vrt'

    def _init_pipeline_folders(self):
        '''Create harmonization directories and empty runs manifest.'''
        super()._init_pipeline_folders()
        if not self.runs_manifest.exists():
            controller.Controller[dict](self.runs_manifest).persist({})

    def validate(self, require_run: bool = False) -> typing.Self:
        '''
        Validate that harmonization directory and manifest exist.

        Args:
            require_run:
                If True, also validates that the effective run directory
                and valid pixel mask exist.

        Returns:
            typing.Self:
                Validated instance for method chaining.

        Raises:
            ArtifactError:
                If expected directory structure or files are missing.
        '''
        super().validate()
        if not self.runs_manifest.is_file():
            raise controller.ArtifactError(
                f'Harmonization runs manifest does not exist '
                f'or is not a file: {self.runs_manifest}'
            )
        if require_run:
            if not self.effective_run_folder.is_dir():
                raise controller.ArtifactError(
                    f'Harmonization run directory does not exist '
                    f'or is not a directory: {self.effective_run_folder}'
                )
            if not self.valid_mask_raster.is_file():
                raise controller.ArtifactError(
                    f'Harmonization valid mask raster does not exist '
                    f'or is not a file: {self.valid_mask_raster}'
                )
        return self
