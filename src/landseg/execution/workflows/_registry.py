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

# pylint: disable=missing-class-docstring
# pylint: disable=too-few-public-methods

'''
Registry of workflow commands and their callable implementations.

Defines the set of valid workflows names and maps each to its execution
function. Provides utilities for validating and retrieving workflows by
name with both runtime checks and static type safety.
'''

# standard imports
import typing
# local imports
import landseg.configs as configs
import landseg.execution.workflows as workflows

# allowed workflow names
WorkflowName = typing.Literal[
    'study-sweep',
]
_ALLOWED = set(typing.get_args(WorkflowName))

# workflow registry
class WorkflowFn(typing.Protocol):
    def __call__(self, config: configs.RootConfig) -> typing.Any: ...

PIPELINES: dict[WorkflowName, WorkflowFn] = {
    'study-sweep': workflows.sweep,
}

# runtime safe access
def get(name: str) -> WorkflowFn:
    '''
    Retrieve the workflow function associated with a given name.

    Args:
        name: Pipeline identifier as a string.

    Returns:
        Callable implementing the workflow.

    Raises:
        KeyError: If the name is not a recognized workflow.
    '''

    def _is_workflow_name(name: str) -> typing.TypeGuard[WorkflowName]:
        # Return True if the input string is a valid workflow name.
        return name in _ALLOWED

    if not _is_workflow_name(name):
        raise KeyError(f'Unknown workflow name: {name}; allowed: {_ALLOWED}')
    return PIPELINES[name]
