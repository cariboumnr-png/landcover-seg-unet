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
Canonical filesystem paths for training and evaluation sessions.

Public APIs:
    - `SessionPaths`: Run-scoped path handle for session runs.
    - `resolve_session_run`: Resolve experiment, results, or run path.
'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base

# ----- private constants
_SESSION_RUN_MARKERS = (
    'step_results.json',
    'evaluation.json',
    'checkpoints',
    'plots',
    'previews',
    'logs',
)


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class SessionPaths(base.RunScopedArtifactPaths):
    '''Session stage and run-bound path handle.'''

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
    def checkpoint_catalog(self) -> pathlib.Path:
        '''Return path to the checkpoint catalog envelope JSON.'''
        return self.checkpoints / 'status.json'

    @property
    def evaluation(self) -> pathlib.Path:
        '''Return path to the session evaluation results JSON.'''
        return self.effective_run_folder / 'evaluation.json'

    @property
    def step_results(self) -> pathlib.Path:
        '''Return path to the step results JSON file.'''
        return self.effective_run_folder / 'step_results.json'

    @property
    def training_curves(self) -> pathlib.Path:
        '''Return path to the rendered training curves PNG file.'''
        return self.plots / 'training_curves.png'

    def preview(self, head: str) -> pathlib.Path:
        '''
        Return path to the rendered preview PNG for a model head.

        Args:
            head:
                Prediction head identifier.

        Returns:
            pathlib.Path:
                File path to `{head}_preview.png`.
        '''
        return self.previews / f'{head}_preview.png'

    def list_previews(self) -> list[pathlib.Path]:
        '''
        Return sorted existing preview PNG paths under this run.

        Returns:
            list[pathlib.Path]:
                Sorted list of existing `*_preview.png` files.
        '''
        if not self.previews.is_dir():
            return []
        return sorted(self.previews.glob('*_preview.png'))

    def per_class_plot(self, head: str) -> pathlib.Path:
        '''
        Return path to the per-class IoU plot PNG for a model head.

        Args:
            head:
                Prediction head identifier.

        Returns:
            pathlib.Path:
                File path to `{head}_per_class_iou.png`.
        '''
        return self.plots / f'{head}_per_class_iou.png'

    def confusion_matrix_plot(self, head: str) -> pathlib.Path:
        '''
        Return path to the confusion matrix plot PNG for a head.

        Args:
            head:
                Prediction head identifier.

        Returns:
            pathlib.Path:
                File path to `{head}_confusion_matrix.png`.
        '''
        return self.plots / f'{head}_confusion_matrix.png'

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

    def _init_pipeline_folders(self) -> None:
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
                If True, also validates that the handle is run-bound
                and its run subdirectories (`checkpoints`, `logs`,
                `plots`, `previews`) exist.

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
                    'Session run directory does not exist '
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


# ----- public functions
def resolve_session_run(
    path: str | pathlib.Path,
    run_id: int | str | pathlib.Path | None = None,
) -> SessionPaths:
    '''
    Resolve a run-bound `SessionPaths` handle from a directory path.

    Accepts an experiment root (containing `results/run_XXXX`), a
    session results root (containing `run_XXXX` folders), or a direct
    run folder, classifying solely by schema markers without recursive
    globbing or directory creation.

    Args:
        path:
            Experiment root, session results root, or run folder path.
        run_id:
            Optional run selector (`int`, digit string, or folder name).
            When `None` on a multi-run root, opens the highest run ID.

    Returns:
        SessionPaths:
            Frozen run-bound handle for the resolved session run.

    Raises:
        ArtifactError:
            If `path` does not exist, matches no recognized session
            layout, or conflicts with `run_id`.
    '''
    target = pathlib.Path(path)
    if not target.is_dir():
        raise controller.ArtifactError(
            f'Session path does not exist or is not a directory: {target}. '
            'Expected an experiment root (<exp>/results/run_XXXX), '
            'a session results root (<results>/run_XXXX), '
            'or a session run folder.'
        )

    if _is_run_folder(target):
        if run_id is not None:
            expected = _format_run_id(run_id)
            if target.name != expected:
                raise controller.ArtifactError(
                    f'Run folder "{target}" does not match requested '
                    f'run_id "{expected}".'
                )
        return SessionPaths.from_run_folder(target)

    if SessionPaths.list_runs(target):
        return SessionPaths.open_run(target, run_id=run_id)

    results_root = target / 'results'
    if results_root.is_dir():
        return SessionPaths.open_run(results_root, run_id=run_id)

    raise controller.ArtifactError(
        f'Unrecognized session directory layout at "{target}". '
        'Expected an experiment root (<exp>/results/run_XXXX), '
        'a session results root (<results>/run_XXXX), '
        'or a session run folder.'
    )


# ----- private helpers
def _is_run_folder(path: pathlib.Path) -> bool:
    '''Return True when `path` exhibits session run folder markers.'''
    if base.RUN_DIR_RE.match(path.name) is not None:
        return True
    return any((path / marker).exists() for marker in _SESSION_RUN_MARKERS)


def _format_run_id(run_id: int | str | pathlib.Path) -> str:
    '''Normalize run selector to canonical folder name.'''
    if isinstance(run_id, bool):
        raise controller.ArtifactError(f'Invalid run identifier: {run_id!r}')
    if isinstance(run_id, int):
        return f'run_{run_id:04d}'
    if isinstance(run_id, str) and run_id.isdigit():
        return f'run_{int(run_id):04d}'
    return pathlib.Path(run_id).name
