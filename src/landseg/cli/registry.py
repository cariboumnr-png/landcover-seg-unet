# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Registry of pipeline commands and their callable implementations.

Defines the set of valid pipeline names and maps each to its execution
function. Provides utilities for validating and retrieving pipelines by
name with both runtime checks and static type safety.
'''

# standard imports
import typing
# local imports
import landseg.cli.pipelines as pipelines
import landseg.configs as configs

# allowed pipeline names
PipelineName = typing.Literal[
    'default',
    'data-ingest',
    'data-prepare',
    'diagnose-overfit',
    'model-evaluate',
    'model-train',
    'study-sweep',
    'study-analysis',
]
_ALLOWED = set(typing.get_args(PipelineName))

# pipeline registry
class PipelineFn(typing.Protocol):
    def __call__(self, config: configs.RootConfig) -> typing.Any: ...

PIPELINES: dict[PipelineName, PipelineFn] = {
    'default': pipelines.default_action,
    'data-ingest': pipelines.ingest,
    'data-prepare': pipelines.prepare,
    'diagnose-overfit': pipelines.overfit,
    'model-evaluate': pipelines.evaluate,
    'model-train': pipelines.train,
    'study-sweep': pipelines.sweep,
    'study-analysis': pipelines.analyze,
}

# runtime safe access
def get_pipeline(name: str) -> PipelineFn:
    '''
    Retrieve the pipeline function associated with a given name.

    Args:
        name: Pipeline identifier as a string.

    Returns:
        Callable implementing the pipeline.

    Raises:
        KeyError: If the name is not a recognized pipeline.
    '''

    def _is_pipeline_name(name: str) -> typing.TypeGuard[PipelineName]:
        # Return True if the input string is a valid pipeline name.
        return name in _ALLOWED

    if not _is_pipeline_name(name):
        raise KeyError(f'Unknown pipeline name: {name}; allowed: {_ALLOWED}')
    return PIPELINES[name]
