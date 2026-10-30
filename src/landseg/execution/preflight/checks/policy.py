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
Execution policy diagnostic checks.

Inspects collision handling and other execution rules that affect
how artifacts are written.

Public APIs:
    - `collision_policy`: Inspect block collision resolution policy.
'''

# local imports
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema


# ----- constants
_VALID_POLICIES = {'skip', 'overwrite', 'error'}


# ----- public functions
def collision_policy(
    ctx: context.InspectionContext,
    pid: str = 'collision_policy',
) -> schema.ProbeResult:
    '''Inspect block collision resolution policy.'''
    policy = ctx.config.data.ingestion.datablocks.collision_policy.lower()

    if policy == 'overwrite':
        status = schema.ProbeStatus.WARN
        message = "Policy 'overwrite' replaces existing blocks"
    elif policy in {'skip', 'error'}:
        status = schema.ProbeStatus.PASS
        message = f"Collision policy '{policy}' configured"
    else:
        return schema.ProbeResult(
            pid=pid,
            category=schema.ProbeCategory.POLICY,
            status=schema.ProbeStatus.FAIL,
            message=f"Unsupported collision policy '{policy}'",
            details={'policy': policy, 'valid': sorted(_VALID_POLICIES)},
        )

    return schema.ProbeResult(
        pid=pid,
        category=schema.ProbeCategory.POLICY,
        status=status,
        message=message,
        details={'policy': policy},
    )
