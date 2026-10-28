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
        self.val.require_string(self.crs_string)
        ox, oy = self.val.require_sequence(self.origin, 2)
        self.val.require_numeric(ox)
        self.val.require_numeric(oy)

        px, py = self.val.require_sequence(self.pixel_size, 2)
        px = self.val.require_numeric_min(px, 0.0)
        py = self.val.require_numeric_min(py, 0.0)
        if px != py:
            raise ValueError('Only square pixels are supported')

        ex, ey = self.val.require_sequence(self.extent_in_crs_units, 2)
        self.val.require_numeric_min(ex, 0.0)
        self.val.require_numeric_min(ey, 0.0)


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
        tx, ty = self.val.require_sequence(self.tile_size, 2)
        tx = self.val.require_numeric_min(tx, 1, int)
        ty = self.val.require_numeric_min(ty, 1, int)
        if tx != ty:
            raise ValueError('Only square blocks are supported.')

        sx, sy = self.val.require_sequence(self.tile_stride, 2)
        sx = self.val.require_numeric_range(sx, (1, tx), int)
        sy = self.val.require_numeric_range(sy, (1, ty), int)
        if sx != sy:
            raise ValueError('Only equal row/column stride is supported.')

        match self.mode:
            case 'ref': pass # defer file system checks to domain validation
            case 'manual': self.manual_specs.validate()
            case _: raise base.ConfigValidationError(
                f'Grid mode can either be "ref" or "manual", got: {self.mode}'
            )
