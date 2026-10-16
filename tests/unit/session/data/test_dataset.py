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

# pylint: disable=protected-access

'''Unit tests for custom torch dataset module (dataset.py).'''

# third-party imports
import numpy
import pytest
import torch
# local imports
import landseg.session.data.dataset as dataset


# ----- `BlockConfig` tests
def test_block_config_properties():
    '''
    Given: Data block size and dataset patch size configs.
    When: Instantiation of `BlockConfig`.
    Then: Successfully access `patch_per_dim` and `patch_per_blk`.
    '''

    cfg = dataset.BlockDatasetConfig(
        block_size=256,
        patch_size=128,
        image_key='image',
        label_key='label',
    )

    assert cfg.patch_per_dim == 2 # 256/128
    assert cfg.patch_per_blk == 4 # (256/128) ^ 2


def test_block_config_invalid_sizes():
    '''
    Given: Incorrect block and patch size config.
    When: Instantiation of `BlockConfig`.
    Then: Raises `ValueError`.
    '''

    with pytest.raises(ValueError, match='Data block size must > '):
        _ = dataset.BlockDatasetConfig(
            block_size=128,
            patch_size=256,
            image_key='image',
            label_key='label',
        )

    with pytest.raises(ValueError, match='Data block size must be divisible '):
        _ = dataset.BlockDatasetConfig(
            block_size=256,
            patch_size=127,
            image_key='image',
            label_key='label',
        )


# ----- `_BlockDataset` tests
def test_block_dataset_instantiation(tmp_path):
    '''
    Given: A `_BlockDataset` object.
    When: After the class object is instantiated.
    Then: Correctly initialize class attributes.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=False)

    assert len(dt) == 4
    numpy.testing.assert_array_equal(
        dt.imgs,
        numpy.array([
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[1.1, 2.1], [3.1, 4.1]]],
            [[[1.2, 2.2], [3.2, 4.2]]],
            [[[1.3, 2.3], [3.3, 4.3]]],
        ]) # [4, 1, 2, 2]
    )
    assert torch.equal(
        dt.domain['domain_1'],
        torch.tensor(10, dtype=torch.long)
    )
    assert torch.equal(
        dt.domain['domain_2'],
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    )


def test_block_dataset_getitem_index_error(tmp_path):
    '''
    Given: A `_BlockDataset` configured with no augment flip.
    When: call `__getitem__` from the class object with invalid key.
    Then: Raise `IndexError`.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=False)

    with pytest.raises(IndexError, match='Invalid patch idx'):
        _, _, _ = dt[4] # valid: [0, 3]


def test_block_dataset_getitem_no_label(tmp_path):
    '''
    Given: A `_BlockDataset` with a placeholder input label array.
    When: call `__getitem__` to access label data.
    Then: Returns the label data as an empty `torch.Tensor`.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=False, label=numpy.array([1]))
    _, y, _ = dt[0]

    assert y.numel() == 0


def test_block_dataset_getitem_no_augment_flip(tmp_path):
    '''
    Given: A `_BlockDataset` configured with no augment flip.
    When: call `__getitem__` from the class object.
    Then: Correctly return the original values from the called key.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=False)
    x, y, domains = dt[0]

    assert torch.equal(
        x,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    ) # image [1, 2, 2]
    assert torch.equal(
        y,
        torch.tensor([[[5, 6], [7, 8]]], dtype=torch.long)
    ) # label [1, 2, 2]
    assert torch.equal(
        domains['domain_1'],
        torch.tensor(10, dtype=torch.long)
    )
    assert torch.equal(
        domains['domain_2'],
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    )


def test_block_dataset_getitem_hflip_only(tmp_path):
    '''
    Given: A `_BlockDataset` configured with augment flip.
    When: manually set hflip only and call `__getitem__`.
    Then: Correctly return horizontal flipped values from the called key.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=True)

    values = iter([
        torch.tensor(0.1),  # < 0.5 horizontal flip
        torch.tensor(0.9),  # > 0.5 no vertical flip
    ])
    patch = pytest.MonkeyPatch()
    patch.setattr('torch.rand', lambda _: next(values))

    x, y, _ = dt[0]

    assert torch.equal(
        x,
        torch.tensor([[[2.0, 1.0], [4.0, 3.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y,
        torch.tensor([[[6, 5], [8, 7]]], dtype=torch.long)
    )


def test_block_dataset_getitem_vflip_only(tmp_path):
    '''
    Given: A `_BlockDataset` configured with augment flip.
    When: manually set vflip only and call `__getitem__`.
    Then: Correctly return vertically flipped values from the called key.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=True)

    values = iter([
        torch.tensor(0.9),  # > 0.5 no horizontal flip
        torch.tensor(0.1),  # < 0.5 vertical flip
    ])
    patch = pytest.MonkeyPatch()
    patch.setattr('torch.rand', lambda _: next(values))

    x, y, _ = dt[0]

    assert torch.equal(
        x,
        torch.tensor([[[3.0, 4.0], [1.0, 2.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y,
        torch.tensor([[[7, 8], [5, 6]]], dtype=torch.long)
    )


def test_block_dataset_getitem_both_flip(tmp_path):
    '''
    Given: A `_BlockDataset` configured with augment flip.
    When: manually set to flip both directions and call `__getitem__`.
    Then: Correctly return double-flipped values from the called key.
    '''
    dt = _test_block_dataset(tmp_path, aug_flip=True)

    values = iter([
        torch.tensor(0.1),  # < 0.5 horizontal flip
        torch.tensor(0.1),  # < 0.5 vertical flip
    ])
    patch = pytest.MonkeyPatch()
    patch.setattr('torch.rand', lambda _: next(values))

    x, y, _ = dt[0]

    assert torch.equal(
        x,
        torch.tensor([[[4.0, 3.0], [2.0, 1.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y,
        torch.tensor([[[8, 7], [6, 5]]], dtype=torch.long)
    )


# ----- `MultiBlockDataset` tests
def test_multiblock_dataset_instantiation_preload(tmp_path):
    '''
    Given: A `MultiBlockDataset` configured with `preload=True`.
    When: Instantiation of `MultiBlockDataset`.
    Then: Correctly set dataset length and preload count attributes.
    '''
    dt = _test_multiblock_dataset(tmp_path, preload=True)

    assert len(dt) == 8
    assert dt.n_preloaded == 2
    assert dt.n_cached == 0


def test_multiblock_dataset_instantiation_streaming(tmp_path):
    '''
    Given: A `MultiBlockDataset` configured with `preload=False`.
    When: Instantiation of `MultiBlockDataset`.
    Then: Correctly set dataset length and cached count attributes.
    '''
    dt = _test_multiblock_dataset(tmp_path, preload=False)

    assert len(dt) == 8
    assert dt.n_preloaded == 0
    assert dt.n_cached == 2


def test_multiblock_dataset_global_to_local(tmp_path):
    '''
    Given: A `MultiBlockDataset` object.
    When: Map global patch index to local block and patch index.
    Then: Correctly return tuple of block index and patch index.
    '''
    dt = _test_multiblock_dataset(tmp_path, preload=False)

    assert dt._global_to_local(0) == (0, 0)
    assert dt._global_to_local(3) == (0, 3)
    assert dt._global_to_local(4) == (1, 0)
    assert dt._global_to_local(7) == (1, 3)


def test_multiblock_dataset_getitem_preload(tmp_path):
    '''
    Given: A `MultiBlockDataset` configured with `preload=True`.
    When: Call `__getitem__` across different data blocks.
    Then: Correctly return image tensor, label tensor, and domain dictionary.
    '''
    dt = _test_multiblock_dataset(tmp_path, preload=True)
    x0, y0, _ = dt[0]
    x4, y4, _ = dt[4]

    assert torch.equal(
        x0,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y0,
        torch.tensor([[[5, 6], [7, 8]]], dtype=torch.long)
    )
    assert torch.equal(
        x4,
        torch.tensor([[[11.0, 12.0], [13.0, 14.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y4,
        torch.tensor([[[5, 6], [7, 8]]], dtype=torch.long)
    )


def test_multiblock_dataset_getitem_streaming(tmp_path):
    '''
    Given: A `MultiBlockDataset` configured with `preload=False`.
    When: Call `__getitem__` across different data blocks.
    Then: Dynamically load blocks to cache and return matching tensors.
    '''
    dt = _test_multiblock_dataset(tmp_path, preload=False)
    x0, y0, _ = dt[0]
    x4, y4, _ = dt[4]

    assert torch.equal(
        x0,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y0,
        torch.tensor([[[5, 6], [7, 8]]], dtype=torch.long)
    )
    assert torch.equal(
        x4,
        torch.tensor([[[11.0, 12.0], [13.0, 14.0]]], dtype=torch.float32)
    )
    assert torch.equal(
        y4,
        torch.tensor([[[5, 6], [7, 8]]], dtype=torch.long)
    )


def test_multiblock_dataset_getitem_no_label(tmp_path):
    '''
    Given: A `MultiBlockDataset` created from blocks with placeholder labels.
    When: Call `__getitem__` to retrieve samples.
    Then: Return label as an empty `torch.Tensor`.
    '''
    fpath = tmp_path / 'unlbl.npz'
    image = numpy.array([[[1.0, 2.0], [3.0, 4.0]]])
    label = numpy.array([1]) # placeholder label
    numpy.savez(fpath, image=image, label=label)

    config = dataset.BlockDatasetConfig(
        block_size=2,
        patch_size=2,
        image_key='image',
        label_key='label'
    )
    src = {'unlbl': fpath}
    dt = dataset.MultiBlockDataset(src, config, preload=False)
    _, y, _ = dt[0]

    assert y.numel() == 0


def test_multiblock_dataset_domain_features(tmp_path):
    '''
    Given: A `MultiBlockDataset` with categorical and vector domains configured.
    When: Retrieve samples using `__getitem__`.
    Then: Correctly populate domain dictionary with formatted tensors.
    '''
    _ = _test_block_dataset(tmp_path, aug_flip=False, name='blk1.npz')
    config = dataset.BlockDatasetConfig(
        block_size=4,
        patch_size=2,
        image_key='image',
        label_key='label',
        ids_domain={'blk1': 42},
        vec_domain={'blk1': [0.5, 0.75]}
    )
    src = {'blk1': tmp_path / 'blk1.npz'}
    dt = dataset.MultiBlockDataset(src, config, preload=True)
    _, _, dom = dt[0]

    assert torch.equal(dom['ids'], torch.tensor(42, dtype=torch.long))
    assert torch.equal(
        dom['vec'],
        torch.tensor([0.5, 0.75], dtype=torch.float32)
    )


def test_multiblock_dataset_cache_eviction(tmp_path):
    '''
    Given: A streaming `MultiBlockDataset` with cache size `blk_cache_num=1`.
    When: Access items belonging to different blocks sequentially.
    Then: Evict oldest cached block to maintain cache size constraint.
    '''
    dt = _test_multiblock_dataset(tmp_path, preload=False, cache_size=1)

    _ = dt[0] # loads block index 0 into cache
    assert 0 in dt.data.img
    assert len(dt.data.img) == 1

    _ = dt[4] # loads block index 1 into cache, evicts block 0
    assert 1 in dt.data.img
    assert 0 not in dt.data.img
    assert len(dt.data.img) == 1


# ----- helper factories
def _test_block_dataset(tmp_path, *, aug_flip: bool, **kwargs):
    '''Return a `_BlockDataset` obj. with valid default attributes.'''
    blk_name = kwargs.get('name', 'data.npz')
    fpath = tmp_path / blk_name
    image = kwargs.get(
        'image',
        numpy.array([[
            [1.0, 2.0, 1.1, 2.1],
            [3.0, 4.0, 3.1, 4.1],
            [1.2, 2.2, 1.3, 2.3],
            [3.2, 4.2, 3.3, 4.3],
        ]]) # [1, 4, 4]
    )
    label = kwargs.get(
        'label',
        numpy.array([[
            [5, 6, 5, 6],
            [7, 8, 7, 8],
            [5, 6, 5, 6],
            [7, 8, 7, 8]
        ]]) # [1, 4, 4]
    )

    numpy.savez(fpath, image=image, label=label)
    context = dataset.BlockDatasetConfig(
        block_size=4,
        patch_size=2,
        image_key='image',
        label_key='label'
    )
    domains = {
        'domain_1': 10,             # catagorical integer domain
        'domain_2': [0.1, 0.2, 0.3] # continuous vector domain
    }
    return dataset._BlockDataset(fpath, context, domains, augment_flip=aug_flip)


def _test_multiblock_dataset(tmp_path, preload: bool, cache_size: int = 16):
    '''Return a `MultiBlockDataset` obj.'''
    # create two temp blocks
    _ = _test_block_dataset(
        tmp_path,
        aug_flip=False,
        name='blk1.npz'
    )
    _ = _test_block_dataset(
        tmp_path,
        aug_flip=False,
        # different values than default
        name='blk2.npz',
        image=numpy.array([[
            [11.0, 12.0, 11.1, 12.1],
            [13.0, 14.0, 13.1, 14.1],
            [11.2, 12.2, 11.3, 12.3],
            [13.2, 14.2, 13.3, 14.3],
        ]]),
        label=numpy.array([[
            [5, 6, 5, 6],
            [7, 8, 7, 8],
            [5, 6, 5, 6],
            [7, 8, 7, 8]
        ]])
    )
    src = {
        'blk1': tmp_path / 'blk1.npz',
        'blk2': tmp_path / 'blk2.npz',
    }
    # config (same as single block)
    config = dataset.BlockDatasetConfig(
        block_size=4,
        patch_size=2,
        image_key='image',
        label_key='label',
    )
    # return instance
    return dataset.MultiBlockDataset(
        src,
        config,
        preload=preload,
        blk_cache_num=cache_size
    )
