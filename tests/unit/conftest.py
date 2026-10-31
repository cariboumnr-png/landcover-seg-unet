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

# pylint: disable=missing-function-docstring

'''Fixtures for testing `landseg.core` module.'''

# standard imports
import pathlib
# third-party imports
import numpy
import pytest
import torch
# local imports
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.blocks.assembler as assembler


# ----- private fixtures
@pytest.fixture(scope='session')
def _shared_block_files(tmp_path_factory):
    '''Build and persist shared test data blocks once per session.'''
    block_dir = tmp_path_factory.mktemp('dataspecs_blocks')
    img = numpy.random.rand(4, 256, 256).astype(numpy.float32)
    lbl = numpy.random.randint(1, 3, size=(2, 256, 256)).astype(numpy.int64)
    label_specs: dict[str, geo_core.CategoricalSpec] = {
        'head_1': {
            'num_cls': 2,
            'ignore_cls': (255,),
            'index_base': 1,
            'class_name': {'1': 'bg', '2': 'fg'},
        },
        'head_2': {
            'num_cls': 3,
            'ignore_cls': (255,),
            'index_base': 1,
            'class_name': {'1': 'c0', '2': 'c1', '3': 'c2'},
        },
    }
    raster_data = assembler.RasterReadOutput(
        image_array=img,
        image_nodata=numpy.nan,
        image_band_map={'red': 0, 'green': 1, 'blue': 2, 'dem': 3},
        image_padded_dem=None,
        label_array=lbl,
        label_nodata=255,
        label_specs=label_specs,
    )
    paths: dict[str, pathlib.Path] = {}
    for name in ('train_block', 'val_block', 'test_block'):
        block = assembler.build_data_block(
            name, raster_data, ignore_index=255
        )
        fpath = block_dir / f'{name}.npz'
        block.save(fpath)
        paths[name] = fpath
    return paths


# ----- public fixtures
@pytest.fixture
def in_memory_dataspecs():
    '''Provide in-memory `DataSpecs` without physical disk blocks.'''
    dummy_dir = pathlib.Path('/virtual/data')
    return core.DataSpecs(
        name="test_dataset",
        mode="default",
        meta=core.Meta(
            blk_bytes=1024,
            test_blks_grid=(1, 1),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=4,
                height_width=256,
                array_key='image',
                band_map={'red': 0, 'green': 1, 'blue': 2, 'dem': 3},
            ),
            label_specs=core.Meta.Label(
                ignore_index=255,
                array_key='label',
            ),
        ),
        heads=core.Heads(
            class_counts={
                'head_1': [100, 200],
                'head_2': [50, 150, 250],
            },
            logits_adjust={
                'head_1': [0.2, 0.1],
                'head_2': [0.1, 0.1, 0.1],
            },
            head_parent={'head_1': None, 'head_2': None},
            head_parent_cls={'head_1': None, 'head_2': None},
        ),
        splits=core.Splits(
            train={'train_block': dummy_dir / 'train_block.npz'},
            val={'val_block': dummy_dir / 'val_block.npz'},
            test={'test_block': dummy_dir / 'test_block.npz'},
        ),
        domains=core.Domains(
            train=core.Domains.Dom(
                ids_domain={'train_block': 1},
                vec_domain={'train_block': [0.1, 0.2]},
            ),
            val=core.Domains.Dom(
                ids_domain={'val_block': 2},
                vec_domain={'val_block': [0.3, 0.4]},
            ),
            test=core.Domains.Dom(
                ids_domain={'test_block': 3},
                vec_domain={'test_block': [0.5, 0.6]},
            ),
            ids_num=3,
            vec_dim=2,
        ),
    )


@pytest.fixture
def dataspecs(_shared_block_files):
    '''Provide fresh `DataSpecs` mapped to pre-built session data blocks.'''
    return core.DataSpecs(
        name="test_dataset",
        mode="default",
        meta=core.Meta(
            blk_bytes=1024,
            test_blks_grid=(1, 1),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=4,
                height_width=256,
                array_key='image',
                band_map={'red': 0, 'green': 1, 'blue': 2, 'dem': 3},
            ),
            label_specs=core.Meta.Label(
                ignore_index=255,
                array_key='label',
            ),
        ),
        heads=core.Heads(
            class_counts={
                'head_1': [100, 200],
                'head_2': [50, 150, 250],
            },
            logits_adjust={
                'head_1': [0.2, 0.1],
                'head_2': [0.1, 0.1, 0.1],
            },
            head_parent={'head_1': None, 'head_2': None},
            head_parent_cls={'head_1': None, 'head_2': None},
        ),
        splits=core.Splits(
            train={'train_block': _shared_block_files['train_block']},
            val={'val_block': _shared_block_files['val_block']},
            test={'test_block': _shared_block_files['test_block']},
        ),
        domains=core.Domains(
            train=core.Domains.Dom(
                ids_domain={'train_block': 1},
                vec_domain={'train_block': [0.1, 0.2]},
            ),
            val=core.Domains.Dom(
                ids_domain={'val_block': 2},
                vec_domain={'val_block': [0.3, 0.4]},
            ),
            test=core.Domains.Dom(
                ids_domain={'test_block': 3},
                vec_domain={'test_block': [0.5, 0.6]},
            ),
            ids_num=3,
            vec_dim=2,
        ),
    )


@pytest.fixture
def mock_model():
    return _MockMultiheadModel()


class _MockMultiheadModel(torch.nn.Module):
    '''Mock multihead model for engine epoch tests.'''
    def __init__(self, spatial_divisor: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, kernel_size=1)
        self.linear = torch.nn.Linear(2, 2)
        self.spatial_divisor = spatial_divisor
        self.active_heads: list[str] | None = None
        self.frozen_heads: list[str] | None = None
        self.logit_adjust_alpha: float = 1.0

    def set_active_heads(self, active_heads: list[str] | None) -> None:
        self.active_heads = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        self.frozen_heads = frozen_heads

    def reset_heads(self) -> None:
        self.active_heads = None
        self.frozen_heads = None

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        self.logit_adjust_alpha = alpha

    def forward(self, x, ids_domain=None, vec_domain=None):
        _ = ids_domain, vec_domain
        if x.dim() == 2:
            return self.linear(x)
        assert self.active_heads is not None
        return {head: self.conv(x) for head in self.active_heads}
