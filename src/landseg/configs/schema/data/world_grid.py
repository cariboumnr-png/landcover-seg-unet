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
    crs_string: str = 'epsg:3161'
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (0.0, 0.0)
    extent_in_crs_units: tuple[float, float] = (0.0, 0.0)

    def validate(self):
        if not bool(re.fullmatch(r'epsg:\d+', self.crs_string, re.I)):
            raise base.ConfigValidationError(
                f'CRS string must be [EPSG:xxxx], got {self.crs_string}'
            )
        px_x, px_y = self.pixel_size
        self.require_type_range(px_x, 'pixel_size_x', float, (0.0, None))
        self.require_type_range(px_y, 'pixel_size_y', float, (0.0, None))
        if px_x != px_y:
            raise ValueError('Only square pixels are supported')

        ex_x, ex_y = self.extent_in_crs_units
        self.require_type_range(ex_x, 'extent_x', float, (0.0, None))
        self.require_type_range(ex_y, 'extent_y', float, (0.0, None))

@dataclasses.dataclass
class WorldGridConfig(base.BaseConfigSection):
    '''Configuration for `geopipe.grid` module.'''
    tile_size: tuple[int, int] = (256, 256)
    tile_stride: tuple[int, int] = (128, 128)
    mode: str = 'ref'
    ref_fpath: str = ''
    manual_specs: GridSpecs = dataclasses.field(default_factory=GridSpecs)
    output_dpath: str = '${execution.exp_root}/artifacts/world_grids'

    def validate(self) -> None:
        # currently we only accept equal row and col sizes and strides
        if self.tile_size[0] != self.tile_size[1]:
            raise ValueError('Only square blocks are supported.')
        if self.tile_size[0] <= 0:
            raise ValueError('Block size must be positive.')

        if self.tile_stride[0] != self.tile_stride[1]:
            raise ValueError('Only equal row/column stride is supported.')
        if self.tile_stride[0] < 0:
            raise ValueError('Block stride must be zero or positive.')

        match self.mode:
            case 'ref':
                self.require_file(self.ref_fpath, 'grid reference raster')

            case 'manual':
                self.manual_specs.validate()

            case _: raise base.ConfigValidationError(
                f'Grid mode can either be "ref" or "manual", got: {self.mode}'
            )
