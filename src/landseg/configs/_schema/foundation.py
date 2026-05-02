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
# pylint: disable=missing-function-docstring

'''
Data foundation schema
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
import re
import typing

# third-party imports
import omegaconf

# alias
field = dataclasses.field

# -------------------------------DATA FOUNDATION-------------------------------
# ----- grid
@dataclasses.dataclass
class _Extent:
    default_input_dpath: str = '${execution.exp_root}/input/extent_reference'
    filename: str = ''
    filepath: str = ''
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    grid_extent: tuple[float, float] | None = None
    grid_shape: tuple[int, int] | None = None

@dataclasses.dataclass
class _TileSpecs:
    size_row: int = 256
    size_col: int = 256
    overlap_row: int = 0
    overlap_col: int = 0

@dataclasses.dataclass
class _Grid:
    mode: str = 'ref'
    crs: str = omegaconf.MISSING
    extent: _Extent = field(default_factory=_Extent)
    tile_specs: _TileSpecs = field(default_factory=_TileSpecs)

    def __post_init__(self):
        if not self.extent.filepath:
            self.extent.filepath = os.path.join(
                self.extent.default_input_dpath, self.extent.filename
            )
        if not _is_resolved(self.crs) or not _is_resolved(self.mode):
            return
        if not bool(re.fullmatch(r'epsg:\d+', self.crs, re.I)):
            raise ValueError('Invalid CRS identifier. Must be [EPSG:....]')
        if self.mode not in {'ref', 'aoi', 'tiles'}:
            raise ValueError(f'Invalid mode: {self.mode}')

    @property
    def tile_specs_tuple(self) -> tuple[int, int, int, int]:
        return dataclasses.astuple(self.tile_specs)

    def validate(self) -> None:
        if not _is_resolved(self.mode):
            return
        if self.mode == 'ref':
            if not os.path.exists(self.extent.filepath):
                raise FileNotFoundError('Mode=ref but ref raster not provided')
        elif self.mode == 'aoi':
            if not all(self.extent.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.extent.grid_extent or ()):
                raise ValueError('Mode=aoi but grid extent has zero(s)')
        elif self.mode == 'tiles':
            if not all(self.extent.pixel_size):
                raise ValueError('Mode=aoi|tiles but pixel size has zero(s)')
            if not all(self.extent.grid_shape or ()):
                raise ValueError('Mode=tiles but grid shape has zero(s)')

# ----- domain
@dataclasses.dataclass
class DomainFile:
    name: str = omegaconf.MISSING
    path: str = ''
    index_base: int = omegaconf.MISSING

@dataclasses.dataclass
class _Domains:
    default_input_dpath: str = '${execution.exp_root}/input/domain_knowledge'
    files: list[DomainFile] = field(default_factory=lambda: [])
    valid_threshold: float = 0.7
    target_variance: float = 0.9

    def validate(self):
        for file in self.files:
            if not file.path:
                file.path = os.path.join(self.default_input_dpath, file.name)
            if not os.path.exists(file.path):
                raise FileNotFoundError(f'Invalid domain raster: {file.path}')

# ----- data
@dataclasses.dataclass
class _FileNames:
    dev_image: str = omegaconf.MISSING
    dev_label: str = omegaconf.MISSING
    test_image: str = omegaconf.MISSING
    test_label: str = omegaconf.MISSING
    config: str = omegaconf.MISSING

@dataclasses.dataclass
class _FilePaths:
    dev_image: str = ''
    dev_label: str = ''
    test_image: str = ''
    test_label: str = ''
    config: str = ''

@dataclasses.dataclass
class _General:
    ignore_index: int = 255
    image_dem_pad: int = 8

@dataclasses.dataclass
class _DataBlocks:
    name: str = omegaconf.MISSING
    default_input_dpath: str = '${execution.exp_root}/input/${foundation.datablocks.name}'
    filenames: _FileNames = field(default_factory=_FileNames)
    filepaths: _FilePaths = field(default_factory=_FilePaths)
    general: _General = field(default_factory=_General)

    def __post_init__(self):
        # compose file paths
        root = self.default_input_dpath
        paths = self.filepaths
        names = self.filenames
        # defer validation until config is composed and resolved
        if not _is_resolved(self.name):
            return
        # dev image
        if not self.filepaths.dev_image:
            paths.dev_image = os.path.join(root, 'dev', names.dev_image)
        # dev label
        if not self.filepaths.dev_label:
            paths.dev_label = os.path.join(root, 'dev', names.dev_label)
        # test image (optional)
        if not self.filepaths.test_image:
            paths.test_image = os.path.join(root, 'test', names.test_image)
        # test label (optional)
        if not self.filepaths.test_label:
            paths.test_label = os.path.join(root, 'test', names.test_label)
        # config JSON
        if not self.filepaths.config:
            paths.config = os.path.join(root, names.config)
        if not self.name:
            raise ValueError('Input data name not provided')

    @property
    def has_test_data(self) -> bool:
        return (
            os.path.isfile(self.filepaths.test_image) and
            os.path.exists(self.filepaths.test_image) and
            os.path.isfile(self.filepaths.test_label) and
            os.path.exists(self.filepaths.test_label)
        )

    def validate(self) -> None:
        def _must_exist(path: str | None, label: str) -> None:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f'invalid {label}: {path}')
        # checks
        _must_exist(self.filepaths.dev_image, 'dev image')
        _must_exist(self.filepaths.dev_label, 'dev label')
        _must_exist(self.filepaths.config, 'config json')

# ----- composite
@dataclasses.dataclass
class DataFoundation:
    grid: _Grid = field(default_factory=_Grid)
    domains: _Domains = field(default_factory=_Domains)
    datablocks: _DataBlocks = field(default_factory=_DataBlocks)
    output_dpath: str = '${execution.exp_root}/artifacts/foundation'

    def validate(self) -> None:
        self.grid.validate()
        self.domains.validate()
        self.datablocks.validate()

# ------------------------------private  function------------------------------
def _is_resolved(value: typing.Any) -> bool:
    '''Return True if not omegaconf.MISSING and not an interpolation.'''
    if value is omegaconf.MISSING:
        return False
    # OmegaConf marks interpolations as strings like '${...}' pre-resolution
    if isinstance(value, str) and value.strip().startswith('${'):
        return False
    return True
