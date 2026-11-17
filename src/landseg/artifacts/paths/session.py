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
Canonical filesystem paths for session experiment training and
evaluation results.
'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- `SessionPaths` definition
@dataclasses.dataclass
class SessionPaths(base.PipelineArtifactsPaths):
    '''Root entry of a training run.'''

    @property
    def checkpoints(self) -> pathlib.Path:
        return self.effective_run_folder / 'checkpoints'

    @property
    def logs(self) -> pathlib.Path:
        return self.effective_run_folder / 'logs'

    @property
    def plots(self) -> pathlib.Path:
        return self.effective_run_folder / 'plots'

    @property
    def previews(self) -> pathlib.Path:
        return self.effective_run_folder / 'previews'

    @property
    def phase_status(self) -> pathlib.Path:
        return self.checkpoints / 'status.json'

    @property
    def evaluation(self) -> pathlib.Path:
        return self.effective_run_folder / 'evaluation.json'

    @property
    def step_results(self) -> pathlib.Path:
        return self.effective_run_folder / 'step_results.json'

    def best_checkpoint(self, name: str) -> pathlib.Path:
        return self.checkpoints / f'{name}_best.pt'

    def last_checkpoint(self, name: str) -> pathlib.Path:
        return self.checkpoints / f'{name}_last.pt'

    def _init_pipeline_folders(self):
        super()._init_pipeline_folders()
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)
        self.previews.mkdir(parents=True, exist_ok=True)

    def validate(self, require_run: bool = False) -> typing.Self:
        '''
        Validate that session directory exists.

        Args:
            require_run:
                If True, also validates that the effective run directory
                and its subdirectories (checkpoints, logs, plots, previews) exist.

        Returns:
            typing.Self: The validated instance for method chaining.

        Raises:
            ArtifactError:
                If expected directory structure is missing.
        '''
        super().validate()
        if require_run:
            if not self.effective_run_folder.is_dir():
                raise controller.ArtifactError(
                    f'Session run directory does not exist '
                    f'or is not a directory: {self.effective_run_folder}'
                )
            for d, name in (
                (self.checkpoints, 'Checkpoints'),
                (self.logs, 'Logs'),
                (self.plots, 'Plots'),
                (self.previews, 'Previews'),
            ):
                if not d.is_dir():
                    raise controller.ArtifactError(
                        f'{name} directory does not exist '
                        f'or is not a directory: {d}'
                    )
        return self
