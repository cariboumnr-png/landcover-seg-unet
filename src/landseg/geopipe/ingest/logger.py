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
    - `IngestionLogger`: Logger tracking ingest progress and reports.
'''

# standard imports
from __future__ import annotations
import datetime
import os
import typing
import uuid
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.ingestion as contracts
import landseg.utils as utils


# ----- public classes
class IngestionLogger(utils.Logger):
    '''
    A specialized Logger wrapper that collects execution metrics and
    persists a structured JSON run report at shutdown.
    '''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''Initialize the IngestionLogger instance.'''
        super().__init__(*args, **kwargs)
        self.summary: contracts.IngestReportSchema | None = None

    def init_summary(
        self,
        *,
        run_id: str = '',
        run_uid: str | None = None,
        harmonization_run_uid: str = '',
        harmonization_run_id: str = '',
        timestamp: str | None = None
    ) -> None:
        '''Initialize the structured run report summary dictionary.'''
        uid = run_uid or f'ingest_{uuid.uuid4().hex[:16]}'
        t = timestamp or datetime.datetime.now().strftime(c.TF_ISO8601)
        self.summary = {
            'run_uid': uid,
            'run_id': run_id,
            'harmonization_run_uid': harmonization_run_uid,
            'harmonization_run_id': harmonization_run_id,
            'timestamp': t,
            'fingerprint': '',
            'status': 'SUCCESS',
            'domain_maps': [],
            'data_blocks': None,
        }

    @property
    def run_uid(self) -> str:
        '''Return current run unique identifier.'''
        if self.summary:
            return self.summary.get('run_uid', '')
        return ''

    def set_fingerprint(self, fingerprint: str) -> None:
        '''Record fingerprint of the inputs and configs of this run.'''
        if self.summary is not None:
            self.summary['fingerprint'] = fingerprint

    def set_harmonization_reference(
        self,
        *,
        run_uid: str,
        run_id: str
    ) -> None:
        '''Set upstream harmonization run reference in summary.'''
        if self.summary is not None:
            self.summary['harmonization_run_uid'] = run_uid
            self.summary['harmonization_run_id'] = run_id

    def add_domain_report(self, report: contracts.DomainMapReport) -> None:
        '''Append a domain layer map report to summary.'''
        if self.summary is not None:
            self.summary['domain_maps'].append(report)

    def set_data_blocks_report(
        self,
        report: contracts.DataBlocksReport
    ) -> None:
        '''Record the data blocks report to summary.'''
        if self.summary is not None:
            self.summary['data_blocks'] = report

    def set_summary_status(
        self,
        status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    ) -> None:
        '''Update the overall run summary status.'''
        if self.summary is not None:
            self.summary['status'] = status

    def update_runs_manifest(
        self,
        manifest_fpath: str,
        run_folder: str,
    ) -> None:
        '''Record or update current run in the ingestion runs manifest.'''
        if self.summary is None:
            return
        uid = self.summary.get('run_uid', '')
        if not uid:
            return
        record: contracts.IngestionRunRecord = {
            'run_uid': uid,
            'run_id': self.summary.get('run_id', ''),
            'harmonization_run_uid': self.summary.get(
                'harmonization_run_uid', ''
            ),
            'harmonization_run_id': self.summary.get(
                'harmonization_run_id', ''
            ),
            'status': self.summary.get('status', 'FAILED'),
            'timestamp': self.summary.get('timestamp', ''),
            'fingerprint': self.summary.get('fingerprint', ''),
            'run_folder': os.path.abspath(run_folder),
        }
        ctrl = artifacts.Controller[dict](manifest_fpath)
        try:
            manifest_data = ctrl.fetch() or {}
        except artifacts.ArtifactError as exc:
            raise ValueError(
                f'Error reading runs manifest at {manifest_fpath}'
            ) from exc
        manifest_data[uid] = record
        ctrl.persist(manifest_data)

    def on_close(self) -> None:
        '''Persist the collected summary JSON report.'''
        if self.summary is not None:
            ctrl = artifacts.Controller(self.log_file)
            ctrl.persist(self.summary)
