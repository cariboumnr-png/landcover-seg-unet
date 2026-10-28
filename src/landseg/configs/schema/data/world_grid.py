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
        self.require_str(self.crs_string, 'CRS (string)')
        self.require_sequence(self.origin, 'origin', 2)

        px, py = self.require_sequence(self.pixel_size, 'pixel_size', 2)
        px = self.require_float_min(px, 'pixel_size_x', 0.0)
        py = self.require_float_min(py, 'pixel_size_y', 0.0)
        if px != py:
            raise ValueError('Only square pixels are supported')

        ex, ey = self.require_sequence(self.extent_in_crs_units, 'extent', 2)
        self.require_float_min(ex, 'extent_x', 0.0)
        self.require_float_min(ey, 'extent_y', 0.0)


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
        tx, ty = self.require_sequence(self.tile_size, 'tile_size', 2)
        tx = self.require_int_min(tx, 'tile_size_x', 1)
        ty = self.require_int_min(ty, 'tile_size_y', 1)
        if tx != ty:
            raise ValueError('Only square blocks are supported.')

        sx, sy = self.require_sequence(self.tile_stride, 'tile_stride', 2)
        sx = self.require_int_range(sx, 'tile_stride_x', (1, tx))
        sy = self.require_int_range(sy, 'tile_stride_y', (1, ty))
        if sx != sy:
            raise ValueError('Only equal row/column stride is supported.')

        match self.mode:
            case 'ref': pass # defer file system checks to domain validation
            case 'manual': self.manual_specs.validate()
            case _: raise base.ConfigValidationError(
                f'Grid mode can either be "ref" or "manual", got: {self.mode}'
            )
