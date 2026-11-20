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
Lightweight parallel execution framework supporting concurrency.

Provides a unified wrapper around ThreadPoolExecutor and
multiprocessing.Pool to simplify concurrent callable execution.

Jobs are expressed as (function, args, kwargs) tuples and are executed
concurrently with optional progress reporting via tqdm.

Public APIs:
    - `ParallelExecutor`: Configurable process and thread task executor.
    - `get_default_worker_count`: Resolve default worker parallelism.
'''

# standard imports
import concurrent.futures
import multiprocessing
import os
import traceback
import typing
# third-party imports
import tqdm


# ----- public functions
def get_default_worker_count() -> int:
    '''
    Resolve default worker concurrency for process pools.

    Checks `LANDSEG_MAX_WORKERS` environment variable, falling back
    to `max(1, cpu_count() - 1)`.

    Returns:
        int:
            Allocated worker concurrency count.
    '''
    env_override = os.environ.get('LANDSEG_MAX_WORKERS')
    if env_override is not None:
        try:
            return max(1, int(env_override))
        except ValueError:
            pass
    return max(1, multiprocessing.cpu_count() - 1)


# module-level default
CORE_N = get_default_worker_count()


# ----- public classes
class ParallelExecutor:
    '''
    A configurable parallel execution utility.

    The executor accepts a list of jobs defined as tuples:
        (function, args, kwargs)

    and returns a list of results preserving execution order.

    Features:
    - Thread-based execution via concurrent.futures.ThreadPoolExecutor
    - Process-based execution via multiprocessing.Pool
    - Automatic error handling per job with structured error output
    - Optional progress bar integration using tqdm
    '''

    def __init__(
        self,
        max_workers: int | None = None,
        use_threads: bool = False,
        show_progress: bool = True,
        *,
        progress_bar_len: int = 100,
        desc: str | None = None,
    ):
        '''
        Initialize the executor.

        Args:
            max_workers:
                Number of workers. Defaults to `CORE_N`.
            use_threads:
                Use threads instead of processes.
            show_progress:
                Show tqdm progress bar.
            progress_bar_len:
                Width of the progress bar in columns.
            desc:
                Optional leading description for progress bar.
        '''
        self.max_workers = (
            max_workers if max_workers is not None else CORE_N
        )
        self.use_threads = use_threads
        self.show_progress = show_progress
        self.ncol = progress_bar_len
        self.desc = desc

    def run(
        self,
        jobs: list,
        desc: str | None = None,
    ) -> list[typing.Any]:
        '''
        Execute jobs in parallel.

        Args:
            jobs:
                List of tuples (func, args, kwargs).
            desc:
                Optional leading description for progress bar.

        Returns:
            list:
                Results from executing all jobs preserving input order.
        '''
        if self.use_threads:
            return self._run_with_threads(jobs, desc=desc)
        return self._run_with_processes(jobs, desc=desc)

    def place_holder(self):
        '''Placeholder public method.'''

    def _run_with_threads(
        self,
        jobs: list,
        desc: str | None = None,
    ) -> list[typing.Any]:
        '''Run jobs using ThreadPoolExecutor with error handling.'''
        with concurrent.futures.ThreadPoolExecutor(self.max_workers) as exe:
            futures = [
                exe.submit(self._safe_call, func, args, kwargs)
                for func, args, kwargs in jobs
            ]
            it = (f.result() for f in futures)
            if self.show_progress:
                return list(
                    tqdm.tqdm(
                        it,
                        total=len(jobs),
                        ncols=self.ncol,
                        desc=desc or self.desc,
                    )
                )
            return list(it)

    def _run_with_processes(
        self,
        jobs: list,
        desc: str | None = None,
    ) -> list[typing.Any]:
        '''Run jobs using multiprocessing Pool with error handling.'''
        with multiprocessing.Pool(self.max_workers) as pool:
            it = pool.imap(self._wrapper_func, jobs)
            if self.show_progress:
                return list(
                    tqdm.tqdm(
                        it,
                        total=len(jobs),
                        ncols=self.ncol,
                        desc=desc or self.desc,
                    )
                )
            return list(it)

    @staticmethod
    def _wrapper_func(job: tuple) -> typing.Any:
        '''Unpack job tuple for imap with error handling.'''
        func, args, kwargs = job
        return ParallelExecutor._safe_call(func, args, kwargs)

    @staticmethod
    def _safe_call(
        func: typing.Callable,
        args: tuple,
        kwargs: dict,
    ) -> typing.Any:
        '''Execute a function safely, capturing errors.'''
        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit) as sys_exc:
            raise sys_exc
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f'Job failed in {func.__name__}: {e}')
            return {'error': str(e), 'traceback': traceback.format_exc()}
