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
Logging utilities for raster harmonization pipeline execution.

This module provides a specialized `Logger` subclass that tracks
harmonization ETL execution progress and serializes structured JSON run
summaries upon completion.

Public APIs:
    - `HarmonizationLogger`: Logger tracking ETL progress and summary.
'''

# standard imports
import datetime
import pathlib
import typing
import uuid
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.harmonization as contracts
import landseg.runtime as runtime


# ----- public classes
class HarmonizationLogger(runtime.Logger):
    '''Logger tracking harmonization ETL progress and report output.'''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''
        Initialize the `HarmonizationLogger` instance.

        Args:
            *args:
                Positional arguments forwarded to `runtime.Logger`.
            **kwargs:
                Keyword arguments forwarded to `runtime.Logger`.
        '''
        super().__init__(*args, **kwargs)
        self.summary: contracts.HarmonizationReportSchema | None = None

    def init_summary(
        self,
        *,
        run_id: str = '',
        run_uid: str | None = None,
        timestamp: str | None = None
    ) -> None:
        '''
        Initialize the structured ETL run report summary.

        Args:
            run_id:
                Optional identifier for the harmonization run.
            run_uid:
                Optional unique run identifier override.
            timestamp:
                Optional ISO-8601 timestamp override.
        '''
        uid = run_uid or f'harmonize_{uuid.uuid4().hex[:16]}'
        t = timestamp or datetime.datetime.now().strftime(c.TF_ISO8601)
        self.summary = {
            'run_uid': uid,
            'run_id': run_id,
            'timestamp': t,
            'fingerprint': '',
            'status': 'SUCCESS',
            'provenance': {},
            'harmonized_sources': {},
            'finalized_rasters': {},
            'valid_mask_raster': '',
            'grid_id': '',
            'grid_fpath': '',
        }

    @property
    def run_uid(self) -> str:
        '''Return current run unique identifier.'''
        if self.summary:
            return self.summary.get('run_uid', '')
        return ''

    def add_source_provenance(
        self, name: str, source_path: pathlib.Path
    ) -> None:
        '''
        Record source file size and modification timestamp provenance.

        Args:
            name:
                Logical layer name key for provenance tracking.
            source_path:
                Path to the raw source raster file.
        '''
        if self.summary is not None and pathlib.Path(source_path).exists():
            stat = pathlib.Path(source_path).stat()
            provenance: contracts.ProvenanceRecord = {
                'path': str(pathlib.Path(source_path).resolve()),
                'size_bytes': stat.st_size,
                'mtime': stat.st_mtime
            }
            self.summary['provenance'][name] = provenance

    def add_harmonized_source(self, name: str, path: pathlib.Path) -> None:
        '''
        Record a harmonized raster layer output path.

        Args:
            name:
                Logical layer name key.
            path:
                Path to the harmonized GeoTIFF output.
        '''
        if self.summary is not None:
            self.summary['harmonized_sources'][name] = str(
                pathlib.Path(path).resolve()
            )

    def add_finalized_raster(self, name: str, path: pathlib.Path) -> None:
        '''
        Record multi-channel feature composite raster path.

        Args:
            name:
                Logical composite role name (e.g., `'features'`).
            path:
                Path to the finalized composite raster or VRT.
        '''
        if self.summary is not None:
            self.summary['finalized_rasters'][name] = str(
                pathlib.Path(path).resolve()
            )

    def set_valid_mask_raster(self, path: pathlib.Path) -> None:
        '''
        Record valid pixel mask raster path.

        Args:
            path:
                Path to the unified valid-pixel mask raster.
        '''
        if self.summary is not None:
            self.summary['valid_mask_raster'] = str(
                pathlib.Path(path).resolve()
            )

    def set_grid_reference(
        self,
        grid_id: str,
        grid_fpath: pathlib.Path
    ) -> None:
        '''
        Record world grid identifier and artifact file path.

        Args:
            grid_id:
                Canonical world grid identifier string.
            grid_fpath:
                Path to the persisted world grid JSON artifact.
        '''
        if self.summary is not None:
            self.summary['grid_id'] = grid_id
            self.summary['grid_fpath'] = str(pathlib.Path(grid_fpath).resolve())

    def set_identity(self, fingerprint: str) -> None:
        '''
        Record fingerprint of the inputs and configs of this run.

        Args:
            fingerprint:
                Deterministic hash digest identifying run inputs.
        '''
        if self.summary is not None:
            self.summary['fingerprint'] = fingerprint

    def set_status(self, status: contracts.StatusLiteral) -> None:
        '''
        Update the overall run summary status.

        Args:
            status:
                Harmonization run status literal.
        '''
        if self.summary is not None:
            self.summary['status'] = status

    def update_runs_manifest(
        self,
        run_manifest_fpath: pathlib.Path,
        run_folder: pathlib.Path,
        dataset_manifest_fpath: str,
    ) -> None:
        '''
        Record or update the harmonization runs manifest.

        Args:
            run_manifest_fpath:
                Path to the harmonization runs manifest JSON.
            run_folder:
                Directory holding outputs of the current run.
            dataset_manifest_fpath:
                Path to the source dataset manifest file.

        Raises:
            ValueError:
                If reading the existing runs manifest fails.
        '''
        if self.summary is None:
            return
        uid = self.summary.get('run_uid', '')
        if not uid:
            return
        record: contracts.HarmonizationRunRecord = {
            'run_uid': uid,
            'run_id': self.summary.get('run_id', ''),
            'run_folder': self.abs_path(run_folder),
            'source_dataset_manifest': self.abs_path(dataset_manifest_fpath),
            'status': self.summary.get('status', 'FAILED'),
            'timestamp': self.summary.get('timestamp', ''),
            'fingerprint': self.summary.get('fingerprint', '')
        }
        Ctrl = artifacts.Controller[dict[str, contracts.HarmonizationRunRecord]]
        run_ctrl = Ctrl(run_manifest_fpath)
        try:
            manifest_data = run_ctrl.fetch(missing_ok=True) or {}
            # manifest.json can be absent
        except artifacts.ArtifactError as e:
            raise ValueError('Error reading runs manifest.json') from e
        manifest_data[uid] = record
        run_ctrl.persist(manifest_data)

    def on_close(self) -> None:
        '''Persist the collected summary JSON report to `log_file`.'''
        if self.summary is not None and self.log_file:
            Ctrl = artifacts.Controller[contracts.HarmonizationReportSchema]
            Ctrl(self.log_file).persist(self.summary)
