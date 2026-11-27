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

'''Canonical filesystem paths for data preparation artifacts.'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PreparationPaths(base.PipelineArtifactsPaths):
    '''Paths for prepared datasets and split artifacts.'''

    @property
    def is_run_bound(self) -> bool:
        '''Return True because preparation paths are root-scoped.'''
        return True

    @property
    def effective_run_folder(self) -> pathlib.Path:
        '''Return root directory as the effective run folder.'''
        return self.root

    @property
    def train_blocks(self) -> pathlib.Path:
        '''Return directory path for prepared training blocks.'''
        return self.root / 'train_blocks'

    @property
    def val_blocks(self) -> pathlib.Path:
        '''Return directory path for prepared validation blocks.'''
        return self.root / 'val_blocks'

    @property
    def test_blocks(self) -> pathlib.Path:
        '''Return directory path for prepared test blocks.'''
        return self.root / 'test_blocks'

    @property
    def splits_source_blocks(self) -> pathlib.Path:
        '''Return path to source block splits manifest JSON.'''
        return self.root / 'block_splits_source.json'

    @property
    def splits_summary(self) -> pathlib.Path:
        '''Return path to block splits summary JSON.'''
        return self.root / 'block_splits_summary.json'

    @property
    def label_stats(self) -> pathlib.Path:
        '''Return path to label statistics JSON.'''
        return self.root / 'label_stats.json'

    @property
    def image_stats(self) -> pathlib.Path:
        '''Return path to image statistics JSON.'''
        return self.root / 'image_stats.json'

    @property
    def splits_prepared_blocks(self) -> pathlib.Path:
        '''Return path to prepared block splits manifest JSON.'''
        return self.root / 'block_splits_prepared.json'

    @property
    def schema(self) -> pathlib.Path:
        '''Return path to the prepared dataset schema JSON.'''
        return self.root / 'schema.json'

    def _init_pipeline_folders(self):
        '''Create preparation root and split block directories.'''
        super()._init_pipeline_folders()
        self.train_blocks.mkdir(parents=True, exist_ok=True)
        self.val_blocks.mkdir(parents=True, exist_ok=True)
        self.test_blocks.mkdir(parents=True, exist_ok=True)

    def validate(self, require_splits: bool = False) -> typing.Self:
        '''
        Validate that preparation artifact directory and schema exist.

        Args:
            require_splits:
                If True, also validates that block split directories
                (`train_blocks`, `val_blocks`, `test_blocks`) and
                prepared splits manifest exist.

        Returns:
            typing.Self:
                Validated instance for method chaining.

        Raises:
            ArtifactError:
                If expected directory structure or files are missing.
        '''
        super().validate()
        if not self.schema.is_file():
            raise controller.ArtifactError(
                f'Preparation schema file does not exist '
                f'or is not a file: {self.schema}'
            )
        if require_splits:
            for d, name in (
                (self.train_blocks, 'Train blocks'),
                (self.val_blocks, 'Validation blocks'),
                (self.test_blocks, 'Test blocks'),
            ):
                if not d.is_dir():
                    raise controller.ArtifactError(
                        f'{name} directory does not exist '
                        f'or is not a directory: {d}'
                    )
            if not self.splits_prepared_blocks.is_file():
                raise controller.ArtifactError(
                    f'Prepared block splits manifest does not exist '
                    f'or is not a file: {self.splits_prepared_blocks}'
                )
        return self
