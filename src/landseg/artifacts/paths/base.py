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
import os
import typing


# ----- public dataclasses
@dataclasses.dataclass
class PipelineArtifactsPaths:
    '''Paths for pipeline specific artifacts.'''
    root: str
    run_folder: str | None = None
    trace_to_last: bool = False
    _run_id: str = ''
    _run_folder: str = ''

    def __post_init__(self):
        '''Compile current run folder.'''
        if not self._run_id:
            i = 1
            while True:
                candidate_id = f'run_{i:04d}'
                candidate_folder = os.path.join(self.root, candidate_id)
                if not os.path.exists(candidate_folder):
                    break
                i += 1
            if self.trace_to_last and i > 1:
                i -= 1
                self._run_id = f'run_{i:04d}'
            else:
                self._run_id = f'run_{i:04d}'
            self._run_folder = os.path.join(self.root, self._run_id)
            self.run_folder = self._run_folder

    @property
    def effective_run_folder(self) -> str:
        '''Return either provided or compiled run folder path.'''
        return self.run_folder if self.run_folder else self._run_folder

    @property
    def run_id(self) -> str:
        '''Return run folder name as the canonical run identifier.'''
        return os.path.basename(self.effective_run_folder)

    def get_run_folder(self, run_id: int | str | None = None) -> str:
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
        if run_id is not None:
            if isinstance(run_id, int):
                folder = os.path.join(self.root, f'run_{run_id:04d}')
            elif isinstance(run_id, str):
                if run_id.isdigit():
                    folder = os.path.join(self.root, f'run_{int(run_id):04d}')
                elif os.path.isdir(run_id):
                    folder = run_id
                elif os.path.isdir(os.path.join(self.root, run_id)):
                    folder = os.path.join(self.root, run_id)
                else:
                    raise FileNotFoundError(
                        f'Run folder does not exist: {run_id}'
                    )
            else:
                raise TypeError(f'Invalid run_id type: {type(run_id)}')

            if not os.path.isdir(folder):
                raise FileNotFoundError(f'Run folder does not exist: {folder}')
            self.run_folder = folder
            return folder

        runs = sorted(
            d for d in os.listdir(self.root)
            if d.startswith('run_')
            and os.path.isdir(os.path.join(self.root, d))
        )

        if not runs:
            raise FileNotFoundError('No run folders found.')

        self.run_folder = os.path.join(self.root, runs[-1])
        return self.run_folder

    def init_pipeline_folders(self) -> typing.Self:
        '''Initialize folder tree specified by the pipeline.'''
        self._init_pipeline_folder()
        return self

    def _init_pipeline_folder(self) -> None: ...
        # leave for the subclass to implement
