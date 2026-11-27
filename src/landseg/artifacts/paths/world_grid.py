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

'''Canonical filesystem paths for world grid artifacts.'''

# standard imports
import dataclasses
import pathlib
import typing
# local imports
import landseg.artifacts.controller as controller
import landseg.artifacts.paths.base as base


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class WorldGridPaths(base.PipelineArtifactsPaths):
    '''Paths for world grid artifacts.'''

    @property
    def is_run_bound(self) -> bool:
        '''Return True because world grid paths are root-scoped.'''
        return True

    @property
    def effective_run_folder(self) -> pathlib.Path:
        '''Return root directory as the effective run folder.'''
        return self.root

    def validate(self) -> typing.Self:
        '''
        Validate that the world grid directory and report exist.

        Returns:
            typing.Self:
                Validated instance for method chaining.

        Raises:
            ArtifactError:
                If root directory or report file does not exist.
        '''
        super().validate()
        if not self.report.is_file():
            raise controller.ArtifactError(
                f'World grid report does not exist '
                f'or is not a file: {self.report}'
            )
        return self
