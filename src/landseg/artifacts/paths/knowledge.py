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

'''Path manager for repository knowledge base artifacts.'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.controller as controller


# ----- `KnowledgePaths` definition
@dataclasses.dataclass(frozen=True)
class KnowledgePaths:
    '''Path manager for knowledge base embeddings and profiles.'''
    root: pathlib.Path = pathlib.Path('knowledge')

    def __post_init__(self):
        object.__setattr__(self, 'root', pathlib.Path(self.root))

    def profile_dpath(self, profile: str) -> pathlib.Path:
        '''
        Return directory path for a knowledge base profile.

        Args:
            profile:
                Knowledge profile identifier.

        Returns:
            pathlib.Path:
                Profile embeddings directory path.
        '''
        return self.root / 'embeddings' / profile

    def similarity_matrix_fpath(self, profile: str) -> pathlib.Path:
        '''
        Return path to the species similarity matrix tensor.

        Args:
            profile:
                Knowledge profile identifier.

        Returns:
            pathlib.Path:
                File path to `species_similarity_matrix.pt`.
        '''
        return self.profile_dpath(profile) / 'species_similarity_matrix.pt'

    def embeddings_fpath(self, profile: str) -> pathlib.Path:
        '''
        Return path to the species embeddings tensor file.

        Args:
            profile:
                Knowledge profile identifier.

        Returns:
            pathlib.Path:
                File path to `species_embeddings.pt`.
        '''
        return self.profile_dpath(profile) / 'species_embeddings.pt'

    def metadata_fpath(self, profile: str) -> pathlib.Path:
        '''
        Return path to the species metadata JSON file.

        Args:
            profile:
                Knowledge profile identifier.

        Returns:
            pathlib.Path:
                File path to `species_metadata.json`.
        '''
        return self.profile_dpath(profile) / 'species_metadata.json'

    def similarity_csv_fpath(self, profile: str) -> pathlib.Path:
        '''
        Return path to the species similarity matrix CSV export.

        Args:
            profile:
                Knowledge profile identifier.

        Returns:
            pathlib.Path:
                File path to `species_similarity_matrix.csv`.
        '''
        return self.profile_dpath(profile) / 'species_similarity_matrix.csv'

    def validate(self, profile: str | None = None) -> typing.Self:
        '''
        Validate knowledge root directory and optional profile files.

        Args:
            profile:
                Optional profile name to validate matrix and metadata
                files for.

        Returns:
            typing.Self:
                Validated instance for method chaining.

        Raises:
            ArtifactError:
                If expected directory structure or files are missing.
        '''
        if not self.root.is_dir():
            raise controller.ArtifactError(
                f'Knowledge root directory does not exist '
                f'or is not a directory: {self.root}'
            )
        if profile is not None:
            prof_dpath = self.profile_dpath(profile)
            if not prof_dpath.is_dir():
                raise controller.ArtifactError(
                    f'Profile directory does not exist '
                    f'or is not a directory: {prof_dpath}'
                )
            if not self.similarity_matrix_fpath(profile).is_file():
                raise controller.ArtifactError(
                    'Species similarity matrix does not exist or is not a file: '
                    f'{self.similarity_matrix_fpath(profile)}'
                )
            if not self.embeddings_fpath(profile).is_file():
                raise controller.ArtifactError(
                    'Species embeddings do not exist or are not a file: '
                    f'{self.embeddings_fpath(profile)}'
                )
            if not self.metadata_fpath(profile).is_file():
                raise controller.ArtifactError(
                    'Species metadata does not exist or is not a file: '
                    f'{self.metadata_fpath(profile)}'
                )
        return self
