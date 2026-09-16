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
Logging utilities for raster harmonization pipeline execution.

This module provides a specialized `Logger` subclass that tracks
harmonization ETL execution progress and serializes structured JSON run
summaries upon completion.

Public APIs:
    - `HarmonizationLogger`: Logger tracking ETL progress and report JSON.
'''

# standard imports
from __future__ import annotations
import datetime
import os
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.utils as utils

if typing.TYPE_CHECKING:
    from .schema import HarmonizationReportSchema, WorldGridReport


# ----- public classes
class HarmonizationLogger(utils.Logger):
    '''
    A specialized `Logger` wrapper that logs raster harmonization
    progress and persists a structured JSON report at shutdown.
    '''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''Initialize the HarmonizationLogger instance.'''
        super().__init__(*args, **kwargs)
        self.summary: HarmonizationReportSchema | None = None

    def init_summary(
        self,
        *,
        run_id: str = '',
        timestamp: str | None = None
    ) -> None:
        '''Initialize the structured ETL run report summary.'''
        t = timestamp or datetime.datetime.now().strftime(c.TF_ISO8601)
        self.summary = {
            'run_id': run_id,
            'timestamp': t,
            'status': 'SUCCESS',
            'provenance': {},
            'harmonized_sources': {},
            'finalized_rasters': {},
            'valid_mask_raster': '',
            'world_grid': None
        }

    def add_source_provenance(self, name: str, source_path: str) -> None:
        '''Record source file size and modification timestamp provenance.'''
        if self.summary is not None and os.path.exists(source_path):
            stat = os.stat(source_path)
            self.summary['provenance'][name] = {
                'path': os.path.abspath(source_path),
                'size_bytes': stat.st_size,
                'mtime': stat.st_mtime
            }

    def add_harmonized_source(self, name: str, path: str) -> None:
        '''Record a harmonized raster layer output path.'''
        if self.summary is not None:
            self.summary['harmonized_sources'][name] = os.path.abspath(path)

    def add_finalized_raster(self, name: str, path: str) -> None:
        '''Record multi-channel feature composite raster path.'''
        if self.summary is not None:
            self.summary['finalized_rasters'][name] = os.path.abspath(path)

    def set_valid_mask_raster(self, path: str) -> None:
        '''Record valid pixel mask raster path.'''
        if self.summary is not None:
            self.summary['valid_mask_raster'] = os.path.abspath(path)

    def set_world_grid_report(self, report: WorldGridReport) -> None:
        '''Record world grid preparation report.'''
        if self.summary is not None:
            self.summary['world_grid'] = report

    def set_summary_status(
        self,
        status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    ) -> None:
        '''Update the overall run summary status.'''
        if self.summary is not None:
            self.summary['status'] = status

    def on_close(self) -> None:
        '''Persist the collected summary JSON report directly to log_file.'''
        if self.summary is not None and self.log_file:
            ctrl = artifacts.Controller(self.log_file)
            ctrl.persist(self.summary)
