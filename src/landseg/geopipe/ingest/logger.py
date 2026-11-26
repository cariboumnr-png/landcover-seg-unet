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
Logging utilities for data ingestion pipeline execution.

This module provides a specialized `Logger` subclass that tracks data
ingestion progress, records sub-stage reports, and writes a structured
JSON summary report on exit.

Public APIs:
    - `IngestionLogger`: Logger tracking ingest progress and reports.
'''

# standard imports
import datetime
import pathlib
import typing
import uuid
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.ingestion as contracts
import landseg.runtime as runtime


# ----- public classes
class IngestionLogger(runtime.Logger):
    '''
    Specialized logger that records structured ingestion reports.

    Collects domain map and data block metrics and persists a summary
    JSON artifact at shutdown.
    '''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''
        Initialize the `IngestionLogger` instance.

        Args:
            *args:
                Positional arguments forwarded to `runtime.Logger`.
            **kwargs:
                Keyword arguments forwarded to `runtime.Logger`.
        '''
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
        '''
        Initialize the structured run report summary dictionary.

        Args:
            run_id:
                Human-readable ingestion run identifier.
            run_uid:
                Optional unique run ID; generated if omitted.
            harmonization_run_uid:
                Unique ID of the upstream harmonization run.
            harmonization_run_id:
                Human-readable ID of the upstream harmonization run.
            timestamp:
                Optional ISO-8601 timestamp string; defaults to now.
        '''
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

    @property
    def run_id(self) -> str:
        '''Return current run identifier.'''
        if self.summary:
            return self.summary.get('run_id', '')
        return ''

    def set_identity(self, fingerprint: str) -> None:
        '''
        Record fingerprint of the inputs and configs of this run.

        Args:
            fingerprint:
                Deterministic run identity hash string.
        '''
        if self.summary is not None:
            self.summary['fingerprint'] = fingerprint

    def set_harmonization_reference(
        self,
        *,
        run_uid: str,
        run_id: str
    ) -> None:
        '''
        Set upstream harmonization run reference in summary.

        Args:
            run_uid:
                Unique ID of the upstream harmonization run.
            run_id:
                Human-readable ID of the upstream harmonization run.
        '''
        if self.summary is not None:
            self.summary['harmonization_run_uid'] = run_uid
            self.summary['harmonization_run_id'] = run_id

    def add_domain(self, report: contracts.DomainMapReport) -> None:
        '''
        Append a domain layer map report to summary.

        Args:
            report:
                Domain map construction report payload.
        '''
        if self.summary is not None:
            self.summary['domain_maps'].append(report)

    def set_data_blocks(self, report: contracts.DataBlocksReport) -> None:
        '''
        Record the data blocks report to summary.

        Args:
            report:
                Data blocks construction report payload.
        '''
        if self.summary is not None:
            self.summary['data_blocks'] = report

    def set_status(self, status: contracts.StatusLiteral) -> None:
        '''
        Update the overall run summary status.

        Args:
            status:
                Terminal run status literal.
        '''
        if self.summary is not None:
            self.summary['status'] = status

    def update_runs_manifest(
        self,
        manifest_fpath: pathlib.Path,
        run_folder: pathlib.Path,
    ) -> None:
        '''
        Record or update current run in the ingestion runs manifest.

        Args:
            manifest_fpath:
                File path to the ingestion runs manifest JSON.
            run_folder:
                Directory path of the current ingestion run.

        Raises:
            ValueError:
                If reading the existing runs manifest fails.
        '''
        if self.summary is None:
            return
        uid = self.summary.get('run_uid', '')
        if not uid:
            return
        record: contracts.IngestionRunRecord = {
            'run_uid': uid,
            'run_id': self.summary['run_id'],
            'harmonization_run_uid': self.summary['harmonization_run_uid'],
            'harmonization_run_id': self.summary['harmonization_run_id'],
            'status': self.summary['status'],
            'timestamp': self.summary['timestamp'],
            'fingerprint': self.summary['fingerprint'],
            'run_folder': str(run_folder.resolve()),
        }
        Ctrl = artifacts.Controller[dict[str, contracts.IngestionRunRecord]]
        ctrl = Ctrl(manifest_fpath)
        try:
            manifest_data = ctrl.fetch(missing_ok=True) or {}
        except artifacts.ArtifactError as exc:
            raise ValueError(
                f'Error reading runs manifest at {manifest_fpath}'
            ) from exc
        manifest_data[uid] = record
        ctrl.persist(manifest_data)

    def on_close(self) -> None:
        '''Persist the collected summary JSON report.'''
        if self.summary is not None:
            Ctrl = artifacts.Controller[contracts.IngestReportSchema]
            Ctrl(self.log_file).persist(self.summary)
