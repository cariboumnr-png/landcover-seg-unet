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

# pylint: disable=missing-function-docstring

'''
TypedDict definitions for data harmonization execution summaries.

This module provides schemas for serializing execution reports, source
provenance, and raster metadata for data harmonization runs.

Public APIs:
    - `ProvenanceRecord`: TypedDict for raw raster file provenance.
    - `HarmonizationPipelineConfig`: Protocol for pipeline configs.
    - `HarmonizationReportSchema`: TypedDict for pipeline run report.
    - `HarmonizationRunRecord`: TypedDict for harmonization run entry.
    - `HarmonizationRunManifest`: Type alias for runs ledger mapping.
'''

# standard imports
from __future__ import annotations
import typing


# ----- public types
class HarmonizationPipelineConfig(typing.Protocol):
    '''Shape of the harmonization pipeline configurations.'''
    @property
    def dataset_manifest(self) -> str: ...
    @property
    def resampling_continuous(self) -> str: ...
    @property
    def resampling_categorical(self) -> str: ...


class HarmonizationRunRecord(typing.TypedDict):
    '''Single harmonization run entry stored in runs manifest.'''
    run_uid: str
    run_id: str
    run_folder: str
    status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    timestamp: str
    fingerprint: str


class HarmonizationReportSchema(typing.TypedDict):
    '''Root report mapping the entire data harmonization pipeline run.'''
    run_uid: str
    run_id: str
    timestamp: str
    fingerprint: str
    status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    provenance: dict[str, ProvenanceRecord]
    harmonized_sources: dict[str, str]
    finalized_rasters: dict[str, str]
    valid_mask_raster: str
    grid_id: str
    grid_fpath: str


class ProvenanceRecord(typing.TypedDict):
    '''Provenance record for a raw source raster file.'''
    path: str
    size_bytes: int
    mtime: float
