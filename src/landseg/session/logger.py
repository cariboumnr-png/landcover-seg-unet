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
Structured session logger with persistent JSON run summaries.

Extends `runtime.Logger` to record session metadata, input counts,
result metrics, and final run status into a `SessionSummary` payload
persisted on close.

Public APIs:
    - `SessionLogger`: Session-scoped logger with JSON run summary.
    - `SessionSummary`: Typed dictionary for persisted run summary.
'''

# standard imports
import datetime
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.runtime as runtime


# ----- public types
SessionStatusLiteral = typing.Literal['RUNNING', 'SUCCESS', 'FAILED']


class SessionSummary(typing.TypedDict):
    '''Typed session summary dictionary.'''
    run_id: str
    pipeline: str
    started_at: str
    status: SessionStatusLiteral
    completed_at: str | None
    inputs: typing.Mapping[str, object]   # liberal dicts for now
    results: typing.Mapping[str, object]  # liberal dicts for now


# ----- public classes
class SessionLogger(runtime.Logger):
    '''Collect session logs and persist structured JSON summaries.'''

    def __init__(self, *arg: typing.Any, **kwargs: typing.Any):
        '''
        Initialize the `SessionLogger` instance.

        Args:
            *arg:
                Positional arguments forwarded to `runtime.Logger`.
            **kwargs:
                Keyword arguments forwarded to `runtime.Logger`.
        '''
        super().__init__(*arg, **kwargs)
        self.summary: SessionSummary | None = None

    def init_summary(
        self,
        *,
        run_id: str,
        command: str,
        timestamp: str | None = None
    ) -> None:
        '''
        Initialize the structured session summary dictionary.

        Args:
            run_id:
                Unique identifier for the current session run.
            command:
                Pipeline or workflow command name.
            timestamp:
                Optional ISO-8601 start timestamp override.
        '''
        t = timestamp or datetime.datetime.now().strftime(c.TF_ISO8601)
        self.summary = {
            'run_id': run_id,
            'pipeline': command,
            'started_at': t,
            'status': 'RUNNING',
            'completed_at': None,
            'inputs': {},
            'results': {}
        }

    def set_inputs(self, inputs: dict[str, typing.Any]):
        '''
        Record inputs of the session.

        Args:
            inputs:
                Mapping of input metadata keys to summary values.
        '''
        if self.summary is not None:
            assert isinstance(self.summary['inputs'], dict) # for typing
            self.summary['inputs'].update(**inputs)

    def set_results(self, results: dict[str, typing.Any]):
        '''
        Record outputs of the session.

        Args:
            results:
                Mapping of result metric keys to summary values.
        '''
        if self.summary is not None:
            assert isinstance(self.summary['results'], dict) # for typing
            self.summary['results'].update(**results)

    def set_status(self, status: SessionStatusLiteral) -> None:
        '''
        Update the overall run summary status.

        Args:
            status:
                Terminal or active status (`'RUNNING'`, `'SUCCESS'`, or
                `'FAILED'`).
        '''
        if self.summary is not None:
            self.summary['status'] = status

    def on_close(self) -> None:
        '''Persist the collected summary JSON report.'''
        t = datetime.datetime.now().strftime(c.TF_ISO8601)
        if self.summary is not None:
            self.summary['completed_at'] = t
            Ctrl = artifacts.Controller[SessionSummary]
            Ctrl(self.log_file).persist(self.summary)
