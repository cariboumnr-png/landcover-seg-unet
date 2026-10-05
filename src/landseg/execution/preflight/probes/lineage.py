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
evaluating `pipeline.validate()` non-destructively.

Public APIs:
    - `probe_lineage`: Evaluate pipeline upstream prerequisites.
'''

# local imports
import landseg.execution.pipelines.base as base
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_lineage(pipeline: base.BasePipeline) -> list[schema.ProbeResult]:
    '''
    Validate upstream pipeline prerequisites and execution lineage.

    Args:
        pipeline:
            concrete pipeline runner instance to validate.

    Returns:
        list[schema.ProbeResult]:
            evaluation result containing passing or failing probe record.
    '''
    try:
        pipeline.validate()
        return [
            schema.ProbeResult(
                probe_id='pipeline_prerequisites',
                category='lineage',
                status=schema.ProbeStatus.PASS,
                message=(
                    f'Prerequisites verified for pipeline '
                    f'"{pipeline.pipeline_name}".'
                ),
            )
        ]
    except Exception as err:  # pylint: disable=broad-exception-caught
        return [
            schema.ProbeResult(
                probe_id='pipeline_prerequisites',
                category='lineage',
                status=schema.ProbeStatus.FAIL,
                message=str(err),
                details={'error_type': type(err).__name__},
            )
        ]
