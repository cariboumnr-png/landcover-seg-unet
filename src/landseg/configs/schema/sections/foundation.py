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
import dataclasses
import os
import re
# local imports
import landseg.configs.schema.utils as utils

# alias
field = dataclasses.field

# -------------------------------DATA FOUNDATION-------------------------------
# ----- grid
@dataclasses.dataclass
class _Extent:
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

    def validate(self):
        # current we only accept equal row and col sizes and strides
        if self.size_row != self.size_col:
            raise ValueError('Only square blocks are supported.')

        if self.overlap_row != self.overlap_col:
            raise ValueError('Only equal row/column stride is supported.')

        if self.size_row <= 0:
            raise ValueError('Block size must be positive.')

        if self.overlap_row < 0:
            raise ValueError('Block stride must be zero or positive.')


@dataclasses.dataclass
class _Grid:
    mode: str = 'ref'
    crs: str = ''
    extent: _Extent = field(default_factory=_Extent)
    tile_specs: _TileSpecs = field(default_factory=_TileSpecs)

    @property
    def tile_specs_tuple(self) -> tuple[int, int, int, int]:
        '''Tile specs in px as (row, col, overlap_row, overlap_col).'''
        return dataclasses.astuple(self.tile_specs)

    def validate(self) -> None:
        # grid mode and corresponding extent configs
        match self.mode:
            case 'ref':
                utils.must_exist(self.extent.filepath, 'extent reference raster')
            case 'aoi':
                if not all(self.extent.pixel_size):
                    raise ValueError('Pixel size has zero(s)')
                if not all(self.extent.grid_extent or ()):
                    raise ValueError('Grid extent has zero(s)')
            case 'tiles':
                if not all(self.extent.pixel_size):
                    raise ValueError('Pixel size has zero(s)')
                if not all(self.extent.grid_shape or ()):
                    raise ValueError('Grid shape has zero(s)')
            case _:
                raise ValueError(f'Invalid grid mode: {self.mode}')
        # crs string format
        if not bool(re.fullmatch(r'epsg:\d+', self.crs, re.I)):
            raise ValueError('Invalid CRS identifier. Must be [EPSG:....]')
        # tile specs
        self.tile_specs.validate()

# ----- domain
@dataclasses.dataclass
class _DomainFile:
    name: str = ''
    path: str = ''
    index_base: int = 1

@dataclasses.dataclass
class _Domains:
    files: list[_DomainFile] = field(default_factory=lambda: [])
    valid_threshold: float = 0.7
    target_variance: float = 0.9

    def add_domain(self, fpath: str, index_base: int):
        utils.must_exist(fpath, 'domain raster')
        if not isinstance(index_base, int):
            raise TypeError(f'Index base must be [int], got {type(index_base)}')
        self.files.append(_DomainFile(path=fpath, index_base=index_base ))

    def validate(self):
        for file in self.files:
            utils.must_exist(file.path, 'domain raster')
            # if domain name is not specified, parse from the path
            if not file.name:
                file.name = os.path.splitext(os.path.basename(file.path))[0]

# ----- data
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
    name: str = ''
    filepaths: _FilePaths = field(default_factory=_FilePaths)
    general: _General = field(default_factory=_General)

    @property
    def has_test_data(self) -> bool:
        return (
            utils.file_exists(self.filepaths.test_image) and
            utils.file_exists(self.filepaths.test_label)
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError('Input data name not provided')
        utils.must_exist(self.filepaths.dev_image, 'model development image raster')
        utils.must_exist(self.filepaths.dev_label, 'model development label raster')
        utils.must_exist(self.filepaths.config, 'data config JSON')

# ----- composite
@dataclasses.dataclass
class DataFoundation:
    grid: _Grid = field(default_factory=_Grid)
    domains: _Domains = field(default_factory=_Domains)
    datablocks: _DataBlocks = field(default_factory=_DataBlocks)
    rebuild: bool = False
    output_dpath: str = (
        '${execution.exp_root}/artifacts/'
        '${foundation.datablocks.name}/foundation'
    )

    def validate(self) -> None:
        self.grid.validate()
        self.domains.validate()
        self.datablocks.validate()
