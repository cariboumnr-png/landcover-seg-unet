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
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or       #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Inspection context for pre-flight diagnostics.

Bundles the resolved root configuration and artifact path tree once so
individual probes stay small and side-effect free. All accessors are
read-only; nothing here creates directories or writes files.

Public APIs:
    - `InspectionContext`: Resolved config + artifact paths for probes.
'''

# standard imports
import dataclasses
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class InspectionContext:
    '''Immutable bundle of configuration and resolved artifact paths.'''
    config: configs.RootConfig
    paths: artifacts.ArtifactPaths

    @classmethod
    def from_config(
        cls,
        config: configs.RootConfig,
        artifact_paths: artifacts.ArtifactPaths | None = None,
    ) -> 'InspectionContext':
        '''Build inspection context from a root configuration.'''
        return cls(
            config=config,
            paths=(
                artifact_paths
                if artifact_paths is not None
                else artifacts.ArtifactPaths.from_config(config)
            ),
        )
