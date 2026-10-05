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
Storage and filesystem diagnostic probes.

Validates output directory access, writability, and path readiness.

Public APIs:
    - `probe_storage`: Validate pipeline output directory access.
'''

# standard imports
import os
# local imports
import landseg.artifacts.paths.base as paths_base
import landseg.execution.pipelines.base as base
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_storage(pipeline: base.BasePipeline) -> list[schema.ProbeResult]:
    '''
    Validate output directory writability for a pipeline.

    Args:
        pipeline:
            pipeline runner instance whose output path is inspected.

    Returns:
        list[schema.ProbeResult]:
            list of storage diagnostic probe results.
    '''
    probes: list[schema.ProbeResult] = []
    p_paths = pipeline.pipeline_paths

    if p_paths is None or not isinstance(
        p_paths, paths_base.PipelineArtifactsPaths
    ):
        return probes

    target_dir = p_paths.root
    check_dir = target_dir
    while check_dir and not os.path.exists(check_dir):
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            break
        check_dir = parent

    if check_dir and os.path.exists(check_dir):
        is_writable = os.access(check_dir, os.W_OK)
        status = (
            schema.ProbeStatus.PASS if is_writable else schema.ProbeStatus.FAIL
        )
        msg = (
            f"'{target_dir}' writable"
            if is_writable
            else f"'{target_dir}' not writable"
        )
    else:
        status = schema.ProbeStatus.PASS
        msg = f"'{target_dir}' path resolved"

    probes.append(
        schema.ProbeResult(
            probe_id='output_directory',
            category='storage',
            status=status,
            message=msg,
            details={'target_dir': target_dir},
        )
    )

    return probes
