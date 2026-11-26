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
Local scratch storage and staging utilities for cloud and cluster runs.

Provides unified discovery of fast local scratch storage (such as NVMe
temp mounts on Databricks clusters) alongside safe context-managed
temporary file lifecycle and staged-publish workflows. Staged writes
build files locally first and publish sequentially to remote filesystems
(e.g., Unity Catalog Volumes), preventing corruption from non-sequential
I/O patterns.

Public APIs:
    - `cleanup_scratch`: Remove obsolete temporary files from scratch.
    - `get_scratch_dir`: Resolve and validate active scratch directory.
    - `scratch_file`: Context-managed temporary file in scratch storage.
    - `stage_to_scratch`: Stage writes before publishing to target.
'''

# standard imports
import contextlib
import os
import pathlib
import shutil
import tempfile
import time
import typing
import uuid


# ----- public functions
def get_scratch_dir(
    scratch_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    '''
    Resolve and validate the active local scratch storage directory.

    Evaluates configured overrides in the following precedence order:
        1. Explicit `scratch_dir` argument.
        2. `LANDSEG_SCRATCH_DIR` environment variable.
        3. System temporary directory (`'landseg_scratch'` under
           `tempfile.gettempdir()`).

    Args:
        scratch_dir:
            Optional explicit directory path to use as scratch.

    Returns:
        pathlib.Path:
            Resolved and validated writable directory path.

    Raises:
        PermissionError:
            If the resolved scratch directory is not writable.
    '''
    if scratch_dir is not None:
        target = pathlib.Path(scratch_dir).expanduser().resolve()
    else:
        env_dir = os.environ.get('LANDSEG_SCRATCH_DIR')
        if env_dir:
            target = pathlib.Path(env_dir).expanduser().resolve()
        else:
            target = (
                pathlib.Path(tempfile.gettempdir()).resolve() /
                'landseg_scratch'
            )

    target.mkdir(parents=True, exist_ok=True)
    if not os.access(target, os.W_OK):
        raise PermissionError(
            f"Scratch directory '{target}' is not writable."
        )

    return target


@contextlib.contextmanager
def scratch_file(
    prefix: str = 'landseg_',
    suffix: str = '',
    scratch_dir: str | pathlib.Path | None = None,
) -> typing.Iterator[pathlib.Path]:
    '''
    Provide a temporary file path inside scratch storage.

    Guarantees cleanup on context exit regardless of execution outcome.

    Args:
        prefix:
            Leading string prefix for temporary filename.
        suffix:
            Trailing extension or suffix for temporary filename.
        scratch_dir:
            Optional explicit scratch directory override.

    Yields:
        pathlib.Path:
            Unique temporary file path within scratch directory.
    '''
    sdir = get_scratch_dir(scratch_dir)
    uid = uuid.uuid4().hex[:12]
    tmp_path = sdir / f'{prefix}{uid}{suffix}'
    try:
        yield tmp_path
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@contextlib.contextmanager
def stage_to_scratch(
    destination: str | pathlib.Path,
    scratch_dir: str | pathlib.Path | None = None,
) -> typing.Iterator[pathlib.Path]:
    '''
    Stage write operations to local scratch before publishing.

    Allocates a unique staging path in scratch storage. If the block
    completes without exception, copies the file sequentially to its
    destination and cleans up the staging file. If an exception occurs,
    discards the staging file and leaves destination untouched.

    Args:
        destination:
            Final destination path for the published file.
        scratch_dir:
            Optional explicit scratch directory override.

    Yields:
        pathlib.Path:
            Temporary staging file path in local scratch storage.
    '''
    dest_path = pathlib.Path(destination).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    sdir = get_scratch_dir(scratch_dir)
    uid = uuid.uuid4().hex[:12]
    staging_path = sdir / f'.stage_{uid}_{dest_path.name}'

    try:
        yield staging_path
        if staging_path.exists():
            shutil.copyfile(staging_path, dest_path)
    finally:
        if staging_path.exists():
            try:
                staging_path.unlink()
            except OSError:
                pass


def cleanup_scratch(
    scratch_dir: str | pathlib.Path | None = None,
    max_age_seconds: float | None = None,
) -> int:
    '''
    Remove lingering temporary files from scratch storage.

    Useful for reclaiming disk space from aborted runs or crashed jobs.

    Args:
        scratch_dir:
            Optional explicit scratch directory override.
        max_age_seconds:
            If specified, only files older than this threshold are
            removed.

    Returns:
        int:
            Count of temporary files successfully removed.
    '''
    sdir = get_scratch_dir(scratch_dir)
    removed_count = 0
    now = time.time()

    for item in sdir.iterdir():
        if not item.is_file():
            continue
        if max_age_seconds is not None:
            try:
                age = now - item.stat().st_mtime
                if age < max_age_seconds:
                    continue
            except OSError:
                continue
        try:
            item.unlink()
            removed_count += 1
        except OSError:
            pass

    return removed_count
