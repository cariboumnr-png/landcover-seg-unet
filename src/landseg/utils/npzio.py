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

'''
Sequential-write helpers for compressed NumPy `.npz` archives.

`numpy.savez_compressed` seeks back into the target file to finalize
zip headers, which append-only storage such as Unity Catalog Volume
FUSE mounts rejects with `OSError: [Errno 5]`. The helpers here build
the archive in memory and write it to the target in one sequential
pass, so the same call works on local, POSIX and Volume storage.

Public APIs:
    - `savez_compressed`: write arrays to a `.npz` file sequentially.
'''

# standard imports
import io
import os
import pathlib
import typing
# third-party imports
import numpy


# ----- public functions
def savez_compressed(
    fpath: pathlib.Path | str,
    **arrays: typing.Any,
) -> None:
    '''
    Write arrays to a compressed `.npz` archive in one sequential pass.

    Drop-in replacement for `numpy.savez_compressed` with keyword
    arrays. The archive is assembled in an in-memory buffer, then
    written to `fpath` with a single write followed by `os.fsync`.

    Args:
        fpath:
            Output file path; must end with `.npz`.
        **arrays:
            Arrays (or array-likes) to store, keyed by archive name.

    Raises:
        ValueError:
            If `fpath` does not end with `.npz`.
    '''
    fpath = pathlib.Path(fpath)
    if fpath.suffix != '.npz':
        raise ValueError(f'Output path must end with .npz: {fpath}')

    buffer = io.BytesIO()
    numpy.savez_compressed(buffer, **arrays)
    with open(fpath, 'wb') as dst:
        dst.write(buffer.getbuffer())
        dst.flush()
        os.fsync(dst.fileno())
