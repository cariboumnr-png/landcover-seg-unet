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

'''Configuration for `geopipe.grid` module.'''

# standard imports
import dataclasses
import re
# local imports
import landseg.configs.schema.base as base


@dataclasses.dataclass
class GridSpecs(base.BaseConfigSection):
    '''Configuration for `geopipe.grid` module.'''
    tile_size: tuple[int, int] = (256, 256)
    tile_stride: tuple[int, int] = (128, 128)
    ref_fpath: str | None = None
    crs_string: str | None = None
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None

    def validate(self):
        # currently we only accept equal row and col sizes and strides
        if self.tile_size[0] != self.tile_size[1]:
            raise ValueError('Only square blocks are supported.')
        if self.tile_size[0] <= 0:
            raise ValueError('Block size must be positive.')

        if self.tile_stride[0] != self.tile_stride[1]:
            raise ValueError('Only equal row/column stride is supported.')
        if self.tile_stride[0] < 0:
            raise ValueError('Block stride must be zero or positive.')


@dataclasses.dataclass
class WorldGridConfig(base.BaseConfigSection):
    '''Configuration for `geopipe.grid` module.'''
    mode: str = 'ref'
    params: GridSpecs = dataclasses.field(default_factory=GridSpecs)
    output_dpath: str = 'experiment/artifacts/world_grids'

    def validate(self) -> None:
        self.params.validate()

        if self.mode == 'ref':
            self.require_file(self.params.ref_fpath, 'grid reference raster')

        elif self.mode == 'manual':
            crs = self.params.crs_string
            if not crs or not bool(re.fullmatch(r'epsg:\d+', crs, re.I)):
                raise ValueError(f'Invalid CRS, must be [EPSG:....], got {crs}')

            if not self.params.origin:
                raise ValueError('Origin not provided')
            if not self.params.pixel_size:
                raise ValueError('Pixel size not provided')
            if not self.params.extent_in_crs_units:
                raise ValueError('Extent (in CRS units) not provided')

            if self.params.pixel_size[0] != self.params.pixel_size[1]:
                raise ValueError('Only square pixels are supported')
