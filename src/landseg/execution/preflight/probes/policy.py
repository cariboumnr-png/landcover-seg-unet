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
Execution policy diagnostic probes.

Inspects collision handling, overwrite permissions, and execution rules.

Public APIs:
    - `collision_policy`: Inspect block collision resolution policy.
'''

# local imports
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def collision_policy(
    root_config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Inspect block collision resolution policy.

    Args:
        root_config:
            root configuration containing ingestion collision policy.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            diagnostic probe record for collision handling policy.
    '''
    policy = root_config.data.ingestion.datablocks.collision_policy.lower()
    probe_id = pid or 'collision_policy'
    if policy == 'overwrite':
        return schema.ProbeResult(
            pid=probe_id,
            category='Policy',
            status=schema.ProbeStatus.WARN,
            message="Policy 'overwrite' replaces existing blocks",
            details={'policy': policy},
        )

    if policy in {'skip', 'error'}:
        return schema.ProbeResult(
            pid=probe_id,
            category='Policy',
            status=schema.ProbeStatus.PASS,
            message=f"Collision policy '{policy}' configured",
            details={'policy': policy},
        )

    return schema.ProbeResult(
        pid=probe_id,
        category='Policy',
        status=schema.ProbeStatus.FAIL,
        message=f"Unsupported collision policy '{policy}'",
        details={'policy': policy},
    )
