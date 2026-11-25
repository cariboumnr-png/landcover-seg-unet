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

'''Unit test suite for `landseg.visual.spatial` module.'''

# standard imports
import pathlib
# third-party imports
import matplotlib
import numpy
import pytest
import torch
# local imports
import landseg.visual as visual

matplotlib.use('Agg')


# ----- Grid Layout Tests
def test_plot_grid_layout_dict(tmp_path: pathlib.Path):
    '''Test plotting grid layout from dict specifications with split mappings.'''
    grid = {'rows': 3, 'cols': 4}
    splits = {
        'train': [(0, 0), (1, 0), (0, 1)],
        'val': [(2, 0), (2, 1)],
        'test': [(3, 2)],
    }
    save_path = tmp_path / 'grid.png'
    fig = visual.plot_grid_layout(
        grid,
        aoi_splits=splits,
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_grid_layout_string_splits(tmp_path: pathlib.Path):
    '''Test plotting grid layout with string-formatted tile identifiers.'''
    grid = {'rows': 2, 'cols': 2}
    splits = {
        'train': ['tile_0_0', 'tile_1_0'],
        'val': ['tile_0_1'],
    }
    save_path = tmp_path / 'grid_str.png'
    fig = visual.plot_grid_layout(
        grid,
        aoi_splits=splits,
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_grid_layout_fallback():
    '''Test plotting grid layout with list-like collection fallback.'''
    tiles = ['a', 'b', 'c', 'd', 'e']
    fig = visual.plot_grid_layout(tiles, show=False)
    assert fig is not None


# ----- Class Frequencies Tests
def test_plot_class_frequencies_dict(tmp_path: pathlib.Path):
    '''Test plotting class pixel frequencies from dictionary.'''
    counts = {
        '0': 150000,
        '1': 45000,
        '2': 2000,  # rare class < 5%
    }
    save_path = tmp_path / 'freq.png'
    fig = visual.plot_class_frequencies(
        counts,
        class_names=['Water', 'Forest', 'Urban'],
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_class_frequencies_empty():
    '''Test plotting class pixel frequencies with empty dictionary fallback.'''
    fig = visual.plot_class_frequencies({}, show=False)
    assert fig is not None


# ----- Spectral Composite Tests
def test_plot_spectral_composite_numpy(tmp_path: pathlib.Path):
    '''Test plotting spectral composite from multi-band numpy array.'''
    raster = numpy.random.rand(4, 32, 32).astype(numpy.float32)
    save_path = tmp_path / 'composite.png'
    fig = visual.plot_spectral_composite(
        raster,
        bands=(2, 1, 0),
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_spectral_composite_channels_last(tmp_path: pathlib.Path):
    '''Test plotting spectral composite when channels are in last dimension.'''
    raster = numpy.random.rand(32, 32, 4).astype(numpy.float32)
    save_path = tmp_path / 'composite_cl.png'
    fig = visual.plot_spectral_composite(
        raster,
        bands=(0, 1, 2),
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_spectral_composite_tensor(tmp_path: pathlib.Path):
    '''Test plotting spectral composite from PyTorch tensor.'''
    raster = torch.from_numpy(numpy.random.rand(4, 32, 32).astype(numpy.float32))
    save_path = tmp_path / 'composite_tensor.png'
    fig = visual.plot_spectral_composite(
        raster,
        bands=(3, 2, 1),
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_spectral_composite_invalid_dim():
    '''Test that 2D array raises ValueError.'''
    raster_2d = numpy.zeros((32, 32), dtype=numpy.float32)
    with pytest.raises(ValueError, match='3D'):
        visual.plot_spectral_composite(raster_2d)


def test_plot_spectral_composite_invalid_bands():
    '''Test that out-of-range band indices raise IndexError.'''
    raster = numpy.zeros((3, 32, 32), dtype=numpy.float32)
    with pytest.raises(IndexError, match='exceed'):
        visual.plot_spectral_composite(raster, bands=(0, 1, 5))
