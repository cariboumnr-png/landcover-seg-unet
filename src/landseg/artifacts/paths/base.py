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
Base class for artifact paths dataclasses for each pipeline.
'''

# standard imports
import dataclasses
import pathlib
import typing


# ----- public dataclasses
@dataclasses.dataclass
class PipelineArtifactsPaths:
    '''Paths for pipeline specific artifacts.'''
    root: pathlib.Path | str
    run_folder: pathlib.Path | str | None = None
    trace_to_last: bool = False
    _run_id: str = ''
    _run_folder: pathlib.Path | str = ''

    def __post_init__(self):
        '''Compile current run folder.'''
        self.root = pathlib.Path(self.root)
        if self.run_folder is not None:
            self.run_folder = pathlib.Path(self.run_folder)
        if not self._run_id:
            i = 1
            root_path = self.root
            while True:
                candidate_id = f'run_{i:04d}'
                candidate_folder = root_path / candidate_id
                if not candidate_folder.exists():
                    break
                i += 1
            if self.trace_to_last and i > 1:
                i -= 1
                self._run_id = f'run_{i:04d}'
            else:
                self._run_id = f'run_{i:04d}'
            self._run_folder = root_path / self._run_id
            self.run_folder = self._run_folder

    @property
    def effective_run_folder(self) -> pathlib.Path:
        '''Return either provided or compiled run folder path.'''
        return (
            pathlib.Path(self.run_folder)
            if self.run_folder
            else pathlib.Path(self._run_folder)
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

    def get_run_folder(
        self,
        run_id: int | str | pathlib.Path | None = None
    ) -> pathlib.Path:
        '''
        Return the path to a run folder.

        Args:
            run_id:
                Integer run ID (e.g. 1 -> run_0001), string run folder
                name/ID (e.g. "run_0001" or "1"), or directory path. If
                None, returns the latest existing run folder.

        Raises:
            FileNotFoundError:
                If the requested run does not exist or no run folders
                exist.
            TypeError:
                If run_id is of an invalid type.
        '''
        root_path = self.root
        if run_id is not None:
            if isinstance(run_id, int):
                folder = root_path / f'run_{run_id:04d}'
            elif isinstance(run_id, (str, pathlib.Path)):
                p = pathlib.Path(run_id)
                if isinstance(run_id, str) and run_id.isdigit():
                    folder = root_path / f'run_{int(run_id):04d}'
                elif p.is_dir():
                    folder = p
                elif (root_path / run_id).is_dir():
                    folder = root_path / run_id
                else:
                    raise FileNotFoundError(
                        f'Run folder does not exist: {run_id}'
                    )
            else:
                raise TypeError(f'Invalid run_id type: {type(run_id)}')

            if not folder.is_dir():
                raise FileNotFoundError(f'Run folder does not exist: {folder}')
            self.run_folder = folder
            return self.run_folder

        if not root_path.exists():
            raise FileNotFoundError('No run folders found.')

        runs = sorted(
            d.name for d in root_path.iterdir()
            if d.name.startswith('run_') and d.is_dir()
        )

        if not runs:
            raise FileNotFoundError('No run folders found.')

        self.run_folder = root_path / runs[-1]
        return self.run_folder

    def init_pipeline_folders(self) -> typing.Self:
        '''Initialize folder tree specified by the pipeline.'''
        self._init_pipeline_folders()
        return self

    def _init_pipeline_folders(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.effective_run_folder.mkdir(parents=True, exist_ok=True)
