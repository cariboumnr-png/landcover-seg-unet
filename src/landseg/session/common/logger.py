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

'''
Subclass wrapper of Logger to handle structured session summaries.
'''

# standard imports
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.utils as utils


class SessionSummary(typing.TypedDict):
    '''Typed session summary dictionary.'''
    run_id: str
    pipeline: str
    started_at: str
    status: typing.Literal['RUNNING', 'SUCCESS', 'FAILED']
    completed_at: str | None
    inputs: typing.Mapping[str, object]   # liberal dicts for now
    results: typing.Mapping[str, object]  # liberal dicts for now


class SessionLogger(utils.Logger):
    '''
    A specialized `Logger` wrapper that collects logging during sessions
    and persists a structured JSON session summary at closing.
    '''

    def __init__(self, *arg: typing.Any, **kwargs: typing.Any):
        '''Initialize the `SessionLogger` instance'''
        super().__init__(*arg, **kwargs)
        self.summary: SessionSummary | None = None

    def init_summary(self, *, run_id: str, pipeline: str, start_time: str):
        '''Initialize the structured session summary dictionary.'''
        self.summary = {
            'run_id': run_id,
            'pipeline': pipeline,
            'started_at': start_time,
            'status': 'RUNNING',
            'completed_at': None,
            'inputs': {},
            'results': {}
        }

    def set_inputs(self, inputs: dict[str, typing.Any]):
        '''Record inputs of the session.'''
        if self.summary is not None:
            assert isinstance(self.summary['inputs'], dict) # for typing
            self.summary['inputs'].update(**inputs)

    def set_results(self, results: dict[str, typing.Any]):
        '''Record outputs of the session.'''
        if self.summary is not None:
            assert isinstance(self.summary['results'], dict) # for typing
            self.summary['results'].update(**results)

    def set_summary_status(
        self,
        status: typing.Literal['RUNNING', 'SUCCESS', 'FAILED']
    ) -> None:
        '''Update the overall run summary status.'''
        if self.summary is not None:
            self.summary['status'] = status

    def on_close(self) -> None:
        '''Persist the collected summary JSON report.'''
        if self.summary is not None:
            ctrl = artifacts.Controller(self.log_file)
            ctrl.persist(self.summary)
