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
#                                   and limitations under the License.        #
# =========================================================================== #

# pylint: disable=protected-access

'''Unit tests for dataloader construction module (loader.py).'''

# standard imports
import dataclasses
# third-party imports
import numpy
import pytest
import torch
# local imports
import landseg.session.data.loader as loader


# ----- `_collate_multi_block` tests
def test_collate_multi_block_labeled():
    '''
    Given: A batch of labeled dataset patch samples.
    When: `_collate_multi_block` is called on the batch.
    Then: Correctly stack image tensors, label tensors, and domain features.
    '''
    sample1 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.ones((128, 128), dtype=torch.long),
        {'dom_1': torch.tensor(10, dtype=torch.long)}
    )
    sample2 = (
        torch.ones((4, 128, 128), dtype=torch.float32) * 2,
        torch.ones((128, 128), dtype=torch.long) * 2,
        {'dom_1': torch.tensor(20, dtype=torch.long)}
    )

    batch = [sample1, sample2]
    xs, ys, doms = loader._collate_multi_block(batch)

    assert xs.shape == (2, 4, 128, 128)
    assert ys.shape == (2, 128, 128)
    assert torch.equal(doms['dom_1'], torch.tensor([10, 20], dtype=torch.long))


def test_collate_multi_block_unlabeled():
    '''
    Given: A batch of unlabeled dataset patch samples with empty label tensors.
    When: `_collate_multi_block` is called on the batch.
    Then: Correctly stack image tensors and empty label tensor.
    '''
    sample1 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.empty(0, dtype=torch.long),
        {'dom_1': torch.tensor(10, dtype=torch.long)}
    )
    sample2 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.empty(0, dtype=torch.long),
        {'dom_1': torch.tensor(20, dtype=torch.long)}
    )

    batch = [sample1, sample2]
    xs, ys, doms = loader._collate_multi_block(batch)

    assert xs.shape == (2, 4, 128, 128)
    assert ys.numel() == 0
    assert torch.equal(doms['dom_1'], torch.tensor([10, 20], dtype=torch.long))


def test_collate_multi_block_inconsistent_y_shape():
    '''
    Given: A batch containing labeled samples with mismatched label shapes.
    When: `_collate_multi_block` is called on the batch.
    Then: Raise `ValueError` for inconsistent label shapes.
    '''
    sample1 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.ones((128, 128), dtype=torch.long),
        {}
    )
    sample2 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.ones((64, 64), dtype=torch.long),
        {}
    )

    with pytest.raises(ValueError, match='inconsistent y shapes'):
        _ = loader._collate_multi_block([sample1, sample2])


def test_collate_multi_block_mixed_labeled_unlabeled():
    '''
    Given: A batch containing a mixture of unlabeled and labeled samples.
    When: `_collate_multi_block` is called on the batch.
    Then: Raise `ValueError` for mixed batch composition.
    '''
    sample1 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.empty(0, dtype=torch.long),
        {}
    )
    sample2 = (
        torch.ones((4, 128, 128), dtype=torch.float32),
        torch.ones((128, 128), dtype=torch.long),
        {}
    )

    with pytest.raises(ValueError, match='mixed labeled/unlabeled batch'):
        _ = loader._collate_multi_block([sample1, sample2])


# ----- `_infer_memory_flags` tests
def test_infer_memory_flags_single_mode(dataspecs):
    '''
    Given: Dataset specs with block byte size equal to 0.
    When: `_infer_memory_flags` is called.
    Then: Return memory flags configured for full preloading.
    '''
    dataspecs.meta.blk_bytes = 0
    flags = loader._infer_memory_flags(dataspecs)

    assert flags.preload_train is True
    assert flags.cache_train == 0
    assert flags.preload_val is True
    assert flags.cache_val == 0


def test_infer_memory_flags_high_memory(dataspecs):
    '''
    Given: Abundant system RAM relative to dataset size.
    When: `_infer_memory_flags` is called with high available bytes.
    Then: Return flags configuring preloading for both train and val splits.
    '''
    dataspecs.meta.blk_bytes = 100_000_000 # 100MB per block
    # available RAM: 10GB
    flags = loader._infer_memory_flags(dataspecs, available_bytes=10_000_000_000)

    assert flags.preload_val is True
    assert flags.preload_train is True


def test_infer_memory_flags_low_memory(dataspecs):
    '''
    Given: Constrained system RAM relative to validation split size.
    When: `_infer_memory_flags` is called with low available bytes.
    Then: Fall back to streaming cache mode without preloading.
    '''
    dataspecs.meta.blk_bytes = 100_000_000 # 100MB per block
    dataspecs.splits.val = {f'b{i}': f'path{i}' for i in range(7)} # 700MB val bytes
    # available RAM: 1GB (val_bytes 700MB > 0.6 * 1GB)
    flags = loader._infer_memory_flags(dataspecs, available_bytes=1_000_000_000)

    assert flags.preload_val is False
    assert flags.preload_train is False
    assert flags.cache_val > 0
    assert flags.cache_train > 0


# ----- `_generate_preview_context` tests
def test_generate_preview_context_valid():
    '''
    Given: Valid square patch count per block and test block grid dimensions.
    When: `_generate_preview_context` is called.
    Then: Correctly return `_PreviewContext` dataclass instance.
    '''
    ctx = loader._generate_preview_context(per_blk=4, test_blks_grid=(2, 3))

    assert ctx.patch_per_blk == 4
    assert ctx.patch_per_dim == 2
    assert ctx.block_columns == 2
    assert ctx.patch_grid_shape == (4, 6)


def test_generate_preview_context_non_square():
    '''
    Given: Non-square patch count per block.
    When: `_generate_preview_context` is called.
    Then: Raise `AssertionError`.
    '''
    with pytest.raises(AssertionError, match='patch_per_blk must be square'):
        _ = loader._generate_preview_context(per_blk=5, test_blks_grid=(2, 2))


# ----- `build_dataloaders` tests
@dataclasses.dataclass
class _DummyLoaderConfig:
    batch_size: int = 2
    patch_size: int = 128


def test_build_dataloaders_default_mode(dataspecs, tmp_path):
    '''
    Given: Dataset specs configured for default mode with valid temp blocks.
    When: `build_dataloaders` is called without logger.
    Then: Return `DataLoaders` container with train, val, and test dataloaders.
    '''
    f1 = _create_dummy_npz(tmp_path / 'block_1.npz')
    f2 = _create_dummy_npz(tmp_path / 'block_2.npz')
    f3 = _create_dummy_npz(tmp_path / 'block_3.npz')

    dataspecs.mode = 'default'
    dataspecs.splits.train = {'block_1': f1}
    dataspecs.splits.val = {'block_2': f2}
    dataspecs.splits.test = {'block_3': f3}

    cfg = _DummyLoaderConfig(batch_size=2, patch_size=128)
    loaders = loader.build_dataloaders(dataspecs, cfg)

    assert loaders.train is not None
    assert loaders.val is not None
    assert loaders.test is not None
    assert loaders.meta.batch_size == 2
    assert loaders.meta.patch_size == 128
    assert loaders.meta.preview_context is not None


def test_build_dataloaders_single_mode(dataspecs, tmp_path):
    '''
    Given: Dataset specs configured for single block mode.
    When: `build_dataloaders` is called.
    Then: Return `DataLoaders` container reusing single block loader for train and val.
    '''
    f1 = _create_dummy_npz(tmp_path / 'block_1.npz')

    dataspecs.mode = 'single'
    dataspecs.splits.train = {'block_1': f1}

    cfg = _DummyLoaderConfig(batch_size=2, patch_size=128)
    loaders = loader.build_dataloaders(dataspecs, cfg)

    assert loaders.train is not None
    assert loaders.val is loaders.train
    assert loaders.test is None
    assert loaders.meta.preview_context is None


def test_build_dataloaders_val_only_mode(dataspecs, tmp_path):
    '''
    Given: Dataset specs configured for val_only mode.
    When: `build_dataloaders` is called.
    Then: Return `DataLoaders` with only validation dataloader populated.
    '''
    f2 = _create_dummy_npz(tmp_path / 'block_2.npz')

    dataspecs.mode = 'val_only'
    dataspecs.splits.val = {'block_2': f2}

    cfg = _DummyLoaderConfig(batch_size=2, patch_size=128)
    loaders = loader.build_dataloaders(dataspecs, cfg)

    assert loaders.train is None
    assert loaders.val is not None
    assert loaders.test is None


def test_build_dataloaders_test_only_mode(dataspecs, tmp_path):
    '''
    Given: Dataset specs configured for test_only mode.
    When: `build_dataloaders` is called.
    Then: Return `DataLoaders` with only test dataloader populated.
    '''
    f3 = _create_dummy_npz(tmp_path / 'block_3.npz')

    dataspecs.mode = 'test_only'
    dataspecs.splits.test = {'block_3': f3}

    cfg = _DummyLoaderConfig(batch_size=2, patch_size=128)
    loaders = loader.build_dataloaders(dataspecs, cfg)

    assert loaders.train is None
    assert loaders.val is None
    assert loaders.test is not None


# ----- helper functions
def _create_dummy_npz(path) -> str:
    '''Create a small valid `.npz` block for dataset tests.'''
    fpath = str(path)
    image = numpy.ones((4, 256, 256), dtype=numpy.float32)
    label = numpy.ones((1, 256, 256), dtype=numpy.int64)
    numpy.savez(fpath, image=image, label=label)
    return fpath
