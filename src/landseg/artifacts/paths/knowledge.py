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
Path manager for repository knowledge base artifacts.
'''

# standard imports
import dataclasses
import pathlib


# ----- `KnowledgePaths` definition
@dataclasses.dataclass
class KnowledgePaths:
    '''Path manager for repository knowledge base embeddings and profiles.'''
    root: str = 'knowledge'

    def profile_dpath(self, profile: str) -> str:
        '''Return path to the profile embedding directory.'''
        return str(pathlib.Path(self.root) / 'embeddings' / profile)

    def similarity_matrix_fpath(self, profile: str) -> str:
        '''Return path to species similarity matrix tensor for profile.'''
        return str(
            pathlib.Path(self.profile_dpath(profile))
            / 'species_similarity_matrix.pt'
        )

    def embeddings_fpath(self, profile: str) -> str:
        '''Return path to species embeddings tensor for profile.'''
        return str(
            pathlib.Path(self.profile_dpath(profile))
            / 'species_embeddings.pt'
        )

    def metadata_fpath(self, profile: str) -> str:
        '''Return path to species metadata JSON for profile.'''
        return str(
            pathlib.Path(self.profile_dpath(profile))
            / 'species_metadata.json'
        )

    def similarity_csv_fpath(self, profile: str) -> str:
        '''Return path to species similarity CSV for profile.'''
        return str(
            pathlib.Path(self.profile_dpath(profile))
            / 'species_similarity_matrix.csv'
        )
