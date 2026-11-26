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

'''Canonical filesystem paths for training and evaluation sessions.'''

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
        '''Return directory path for saved model checkpoints.'''
        return self.effective_run_folder / 'checkpoints'

    @property
    def logs(self) -> pathlib.Path:
        '''Return directory path for training and runtime logs.'''
        return self.effective_run_folder / 'logs'

    @property
    def plots(self) -> pathlib.Path:
        '''Return directory path for generated diagnostic plots.'''
        return self.effective_run_folder / 'plots'

    @property
    def previews(self) -> pathlib.Path:
        '''Return directory path for rendered sample previews.'''
        return self.effective_run_folder / 'previews'

    @property
    def phase_status(self) -> pathlib.Path:
        '''Return path to the training phase status JSON.'''
        return self.checkpoints / 'status.json'

    @property
    def evaluation(self) -> pathlib.Path:
        '''Return path to the session evaluation results JSON.'''
        return self.effective_run_folder / 'evaluation.json'

    @property
    def step_results(self) -> pathlib.Path:
        '''Return path to the step results JSON file.'''
        return self.effective_run_folder / 'step_results.json'

    def best_checkpoint(self, name: str) -> pathlib.Path:
        '''
        Return path to the best checkpoint file for a phase.

        Args:
            name:
                Checkpoint prefix or phase identifier.

        Returns:
            pathlib.Path:
                File path to `{name}_best.pt`.
        '''
        return self.checkpoints / f'{name}_best.pt'

    def last_checkpoint(self, name: str) -> pathlib.Path:
        '''
        Return path to the latest checkpoint file for a phase.

        Args:
            name:
                Checkpoint prefix or phase identifier.

        Returns:
            pathlib.Path:
                File path to `{name}_last.pt`.
        '''
        return self.checkpoints / f'{name}_last.pt'

    def _init_pipeline_folders(self):
        '''Create session subdirectories under the run folder.'''
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
                If True, also validates that the effective run
                directory and its subdirectories (`checkpoints`,
                `logs`, `plots`, `previews`) exist.

        Returns:
            typing.Self:
                Validated instance for method chaining.

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
