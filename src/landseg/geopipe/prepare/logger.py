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
Structured execution logging for dataset preparation workflows.

Provides a specialized Logger wrapper that collects execution metrics
from partitioning, normalization, and schema generation stages,
persisting a structured JSON run report upon closure.

Public APIs:
    - `PreparationLogger`: Logger collecting prepare execution reports.
'''

# standard imports
from __future__ import annotations
import datetime
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.preparation as contracts
import landseg.utils as utils


# ----- public classes
class PreparationLogger(utils.Logger):
    '''
    Specialized logger that records structured preparation reports.

    Collects partition, normalization, and schema stage metrics and
    persists a summary JSON artifact upon closure.
    '''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''Initialize the PreparationLogger instance.'''
        super().__init__(*args, **kwargs)
        self.summary: contracts.PreparationReportSchema | None = None

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
            'data_partition': None,
            'normalization': None,
            'schema': None
        }

    def set_data_partition_report(
        self,
        report: contracts.DataPartitionReport
    ) -> None:
        '''Record the data partition report to summary.'''
        if self.summary is not None:
            self.summary['data_partition'] = report

    def set_normalization_report(
        self,
        report: contracts.NormalizationReport
    ) -> None:
        '''Record the normalization report to summary.'''
        if self.summary is not None:
            self.summary['normalization'] = report

    def set_schema_report(
        self,
        report: contracts.SchemaReport
    ) -> None:
        '''Record the schema generation report to summary.'''
        if self.summary is not None:
            self.summary['schema'] = report

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
