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

'''Unit tests for sequential `.npz` writing in npzio.py.'''

# standard imports
import pathlib
import typing
# third-party imports
import numpy
import pytest
# local imports
import landseg.utils.npzio as npzio


# ----- `savez_compressed` round trip
@pytest.mark.parametrize('as_str', [False, True])
def test_savez_compressed_round_trips_arrays(tmp_path, as_str):
    '''Given keyword arrays, numpy.load restores identical arrays.'''
    fpath = tmp_path / 'block.npz'
    image = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    label = numpy.ones((3, 4), dtype=numpy.uint8)
    npzio.savez_compressed(
        str(fpath) if as_str else fpath,
        image=image,
        label=label,
        manifest_json='{"a":1}',
    )
    with numpy.load(fpath) as loaded:
        assert sorted(loaded.files) == ['image', 'label', 'manifest_json']
        numpy.testing.assert_array_equal(loaded['image'], image)
        numpy.testing.assert_array_equal(loaded['label'], label)
        assert loaded['manifest_json'].item() == '{"a":1}'


def test_savez_compressed_overwrites_existing_file(tmp_path):
    '''Given an existing archive, replace its contents entirely.'''
    fpath = tmp_path / 'block.npz'
    npzio.savez_compressed(fpath, old=numpy.zeros(100_000))
    npzio.savez_compressed(fpath, new=numpy.arange(3))
    with numpy.load(fpath) as loaded:
        assert loaded.files == ['new']
        numpy.testing.assert_array_equal(loaded['new'], numpy.arange(3))


# ----- `savez_compressed` path validation
@pytest.mark.parametrize('name', ['block', 'block.npy', 'block.npz.tmp'])
def test_savez_compressed_rejects_non_npz_suffix(tmp_path, name):
    '''Given a path not ending with .npz, raise and write nothing.'''
    fpath = tmp_path / name
    with pytest.raises(ValueError, match='must end with .npz'):
        npzio.savez_compressed(fpath, x=numpy.zeros(2))
    assert not any(tmp_path.iterdir())


# ----- `savez_compressed` sequential write contract
def test_savez_compressed_never_seeks_target(tmp_path, monkeypatch):
    '''Given a target that rejects seek, write a valid archive.'''
    fpath = tmp_path / 'block.npz'
    monkeypatch.setattr(npzio, 'open', _open_sequential_only, raising=False)
    npzio.savez_compressed(fpath, x=numpy.arange(10))
    with numpy.load(fpath) as loaded:
        numpy.testing.assert_array_equal(loaded['x'], numpy.arange(10))


def test_numpy_savez_compressed_fails_on_sequential_target(tmp_path):
    '''Given a target that rejects seek, numpy itself raises OSError.'''
    with _open_sequential_only(tmp_path / 'block.npz', 'wb') as dst:
        with pytest.raises(OSError):
            numpy.savez_compressed(dst, x=numpy.arange(10))


# ----- private helpers
class _SequentialOnlyFile:
    '''Append-only file that mimics Volume FUSE seek behaviour.'''

    def __init__(self, fpath: pathlib.Path):
        self._fh = open(fpath, 'wb')  # pylint: disable=consider-using-with

    def __enter__(self) -> '_SequentialOnlyFile':
        return self

    def __exit__(self, *_: typing.Any) -> None:
        self._fh.close()

    def read(self, *_: typing.Any) -> bytes:
        '''Reject reads; the target is write-only.'''
        raise OSError(5, 'Input/output error')

    def write(self, data: typing.Any) -> int:
        '''Append bytes to the underlying file.'''
        return self._fh.write(data)

    def flush(self) -> None:
        '''Flush the underlying file.'''
        self._fh.flush()

    def fileno(self) -> int:
        '''Return the underlying file descriptor.'''
        return self._fh.fileno()

    def seek(self, offset: int, whence: int = 0) -> int:
        '''Allow no-op seeks only; reject jumps like Volume FUSE.'''
        if whence != 0 or offset != self._fh.tell():
            raise OSError(5, 'Input/output error')
        return offset

    def tell(self) -> int:
        '''Report the current write position.'''
        return self._fh.tell()


def _open_sequential_only(
    fpath: pathlib.Path,
    mode: str,
) -> _SequentialOnlyFile:
    '''Open `fpath` for sequential binary writing only.'''
    assert mode == 'wb'
    return _SequentialOnlyFile(fpath)
