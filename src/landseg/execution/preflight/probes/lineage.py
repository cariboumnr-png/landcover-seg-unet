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
Lineage diagnostic probe implementation.

Validates upstream artifact prerequisites and dependency readiness by
evaluating target prerequisites non-destructively.

Public APIs:
    - `probe_lineage`: Evaluate pipeline upstream prerequisites.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines.base as base
import landseg.execution.preflight.prerequisites as prerequisites
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_lineage(
    target: base.BasePipeline | str,
    root_config: configs.RootConfig | None = None,
) -> list[schema.ProbeResult]:
    '''
    Validate upstream pipeline prerequisites and execution lineage.

    Args:
        target:
            concrete pipeline runner instance or execution target string.
        root_config:
            optional hydra-composed root configuration instance.

    Returns:
        list[schema.ProbeResult]:
            evaluation result containing passing or failing probe records.
    '''
    if isinstance(target, base.BasePipeline):
        target_name = target.pipeline_name
        eff_config = root_config or target.config
        artifact_paths = target.artifact_paths
    else:
        target_name = target
        eff_config = root_config or configs.RootConfig()
        artifact_paths = artifacts.ArtifactPaths.from_config(eff_config)

    checks = prerequisites.check_target_prerequisites(
        target_name, artifact_paths, eff_config
    )
    if checks:
        return [check.to_probe_result() for check in checks]

    if target_name == 'world-grid':
        return [
            schema.ProbeResult(
                probe_id='pipeline_prerequisites',
                category='lineage',
                status=schema.ProbeStatus.PASS,
                message=(
                    'Root pipeline "world-grid" has no upstream '
                    'prerequisites.'
                ),
            )
        ]

    return [
        schema.ProbeResult(
            probe_id='pipeline_prerequisites',
            category='lineage',
            status=schema.ProbeStatus.PASS,
            message=(
                f'No upstream prerequisites defined for target '
                f'"{target_name}".'
            ),
        )
    ]
