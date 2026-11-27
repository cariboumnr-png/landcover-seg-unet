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
Base abstractions for stage and run-scoped artifact path handles.

Public APIs:
    - `RUN_DIR_RE`: Canonical run folder naming regular expresion pattern.
    - `PipelineArtifactsPaths`: Base path handler for a pipeline stage.
    - `RunScopedArtifactPaths`: Base path handler for run-based pipeline.
'''

# standard imports
import dataclasses
import pathlib
import re
import typing
# local imports
import landseg.artifacts.controller as controller

# ----- public constants
RUN_DIR_RE = re.compile(r'^run_(\d+)$')


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PipelineArtifactsPaths:
    '''
    Base frozen artifact path handle for a pipeline stage.

    Constructing directly or via `from_root` binds only `root` without
    filesystem probing or run allocation.
    '''
    root: pathlib.Path
    run_folder: pathlib.Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'root', pathlib.Path(self.root))
        if self.run_folder is not None:
            object.__setattr__(
                self, 'run_folder', pathlib.Path(self.run_folder)
            )

    @classmethod
    def from_root(cls, root: str | pathlib.Path) -> typing.Self:
        '''
        Return an unbound stage path handle without filesystem writes.

        Args:
            root:
                Stage output root directory path.

        Returns:
            typing.Self:
                Stage path handle bound to `root`.
        '''
        return cls(root=pathlib.Path(root))

    @property
    def is_run_bound(self) -> bool:
        '''Return True when handle points to a concrete run folder.'''
        return self.run_folder is not None

    @property
    def effective_run_folder(self) -> pathlib.Path:
        '''
        Return the bound run folder for this handle.

        Returns:
            pathlib.Path:
                Concrete directory holding outputs for this run.

        Raises:
            ArtifactError:
                If the handle is unbound (no `run_folder` attached).
        '''
        if self.run_folder is not None:
            return self.run_folder
        raise controller.ArtifactError(
            f'{type(self).__name__} handle is not bound to a run folder; '
            'use open_run(), from_run_folder(), or allocate_run().'
        )

    @property
    def run_id(self) -> str:
        '''Return run folder name as the canonical run identifier.'''
        return self.effective_run_folder.name

    @property
    def report(self) -> pathlib.Path:
        '''Return the file path of the pipeline execution report.'''
        return self.effective_run_folder / 'report.json'

    @property
    def config(self) -> pathlib.Path:
        '''Return path to persisted configuration JSON.'''
        return self.effective_run_folder / 'config.json'

    def init_pipeline_folders(self) -> typing.Self:
        '''
        Initialize folder tree specified by the pipeline.

        Returns:
            typing.Self:
                Current paths instance for method chaining.
        '''
        self._init_pipeline_folders()
        return self

    def _init_pipeline_folders(self) -> None:
        '''Create root and effective run directories if missing.'''
        self.root.mkdir(parents=True, exist_ok=True)
        self.effective_run_folder.mkdir(parents=True, exist_ok=True)

    def validate(self) -> typing.Self:
        '''
        Defensively validate that the artifact root directory exists.

        Returns:
            typing.Self:
                Validated instance for method chaining.

        Raises:
            ArtifactError:
                If the root directory does not exist or is not a
                directory.
        '''
        if not self.root.exists() or not self.root.is_dir():
            raise controller.ArtifactError(
                f'Root does not exist or is not a directory: {self.root}'
            )
        return self


@dataclasses.dataclass(frozen=True)
class RunScopedArtifactPaths(PipelineArtifactsPaths):
    '''Base handle for stages organized into `run_XXXX` folders.'''

    @classmethod
    def list_runs(cls, root: str | pathlib.Path) -> list[str]:
        '''
        List existing `run_XXXX` folder names sorted by numeric run ID.

        Performs read-only directory inspection with no side effects.

        Args:
            root:
                Stage output root directory to inspect.

        Returns:
            list[str]:
                Sorted run directory names (oldest to newest by ID).
        '''
        root_path = pathlib.Path(root)
        if not root_path.is_dir():
            return []
        matched: list[tuple[int, str]] = []
        for entry in root_path.iterdir():
            if not entry.is_dir():
                continue
            m = RUN_DIR_RE.match(entry.name)
            if m is not None:
                matched.append((int(m.group(1)), entry.name))
        matched.sort(key=lambda item: (item[0], item[1]))
        return [name for _, name in matched]

    @classmethod
    def open_run(
        cls,
        root: str | pathlib.Path,
        run_id: int | str | pathlib.Path | None = None,
    ) -> typing.Self:
        '''
        Open an existing run folder without creating directories.

        Args:
            root:
                Stage output root directory.
            run_id:
                Integer index, numeric string, folder name, or path. If
                `None`, selects the highest run ID under `root`.

        Returns:
            typing.Self:
                Frozen run-bound handle for the resolved run folder.

        Raises:
            ArtifactError:
                If `root`, any run folders, or the requested run folder
                cannot be found.
            TypeError:
                If `run_id` is of an unsupported type.
        '''
        root_path = pathlib.Path(root)
        if not root_path.is_dir():
            raise controller.ArtifactError(
                f'Stage root does not exist or is not a directory: {root_path}'
            )

        if run_id is None:
            runs = cls.list_runs(root_path)
            if not runs:
                raise controller.ArtifactError(
                    f'No run folders found under: {root_path}'
                )
            folder = root_path / runs[-1]
        elif isinstance(run_id, int) and not isinstance(run_id, bool):
            folder = root_path / f'run_{run_id:04d}'
        elif isinstance(run_id, (str, pathlib.Path)):
            if isinstance(run_id, str) and run_id.isdigit():
                folder = root_path / f'run_{int(run_id):04d}'
            else:
                p = pathlib.Path(run_id)
                folder = p if p.is_dir() else (root_path / p)
        else:
            raise TypeError(f'Invalid run_id type: {type(run_id)}')

        if not folder.is_dir():
            raise controller.ArtifactError(
                f'Run folder does not exist: {folder}'
            )
        return cls(root=root_path, run_folder=folder)

    @classmethod
    def from_run_folder(cls, run_dpath: str | pathlib.Path) -> typing.Self:
        '''
        Bind a handle directly to an existing concrete run folder.

        Args:
            run_dpath:
                Existing run directory path.

        Returns:
            typing.Self:
                Frozen run-bound handle for `run_dpath`.

        Raises:
            ArtifactError:
                If `run_dpath` does not exist or is not a directory.
        '''
        folder = pathlib.Path(run_dpath)
        if not folder.is_dir():
            raise controller.ArtifactError(
                f'Run folder does not exist or is not a directory: {folder}'
            )
        return cls(root=folder.parent, run_folder=folder)

    @classmethod
    def allocate_run(cls, root: str | pathlib.Path) -> typing.Self:
        '''
        Allocate the next sequential `run_XXXX` folder for writing.

        Args:
            root:
                Stage output root directory.

        Returns:
            typing.Self:
                Frozen run-bound handle pointing to the newly allocated
                run directory.
        '''
        root_path = pathlib.Path(root)
        runs = cls.list_runs(root_path)
        next_idx = (_parse_run_index(runs[-1]) + 1) if runs else 1
        folder = root_path / f'run_{next_idx:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        return cls(root=root_path, run_folder=folder)


# ----- private helpers
def _parse_run_index(run_name: str) -> int:
    '''Extract integer index from a canonical `run_XXXX` name.'''
    m = RUN_DIR_RE.match(run_name)
    if m is None:
        raise controller.ArtifactError(f'Invalid run folder name: {run_name}')
    return int(m.group(1))
