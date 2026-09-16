# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Logging utilities for data ingestion pipeline execution.

This module provides a specialized `Logger` subclass that tracks data
ingestion progress, records sub-stage reports, and writes a structured
JSON summary report on exit.

Public APIs:
    - `IngestionLogger`: Logger tracking ingest execution and report JSON.
'''

# standard imports
from __future__ import annotations
import datetime
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.utils as utils

if typing.TYPE_CHECKING:
    from .schema import (
        DomainMapReport,
        DataBlocksReport,
        IngestReportSchema,
    )


# ----- public classes
class IngestionLogger(utils.Logger):
    '''
    A specialized Logger wrapper that collects execution metrics and
    persists a structured JSON run report at shutdown.
    '''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''Initialize the IngestionLogger instance.'''
        super().__init__(*args, **kwargs)
        self.summary: IngestReportSchema | None = None

    def init_summary(
        self,
        *,
        run_id: str = '',
        timestamp: str | None = None
    ) -> None:
        '''Initialize the structured run report summary dictionary.'''
        t = timestamp or datetime.datetime.now().strftime(c.TF_ISO8601)
        self.summary = {
            'run_id': run_id,
            'timestamp': t,
            'status': 'SUCCESS',
            'domain_maps': [],
            'data_blocks': None
        }

    def add_domain_report(self, report: DomainMapReport) -> None:
        '''Append a domain layer map report to summary.'''
        if self.summary is not None:
            self.summary['domain_maps'].append(report)

    def set_data_blocks_report(
        self,
        report: DataBlocksReport
    ) -> None:
        '''Record the data blocks report to summary.'''
        if self.summary is not None:
            self.summary['data_blocks'] = report

    def set_summary_status(
        self,
        status: typing.Literal['SUCCESS', 'FAILED']
    ) -> None:
        '''Update the overall run summary status.'''
        if self.summary is not None:
            self.summary['status'] = status

    def on_close(self) -> None:
        '''Persist the collected summary JSON report.'''
        if self.summary is not None:
            ctrl = artifacts.Controller(self.log_file)
            ctrl.persist(self.summary)
