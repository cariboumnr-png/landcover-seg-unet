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
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Lightweight parallel execution framework supporting threads and processes.

This module provides a unified wrapper around Python's ThreadPoolExecutor
and multiprocessing.Pool to simplify parallel execution of callables.

Jobs are expressed as (function, args, kwargs) tuples and are executed
concurrently with optional progress reporting via tqdm.

Key features:
- Thread or process-based execution selectable at runtime
- Safe function execution with error capture per job
- Optional tqdm progress tracking
- Uniform job abstraction across concurrency backends
'''

# standard imports
import concurrent.futures
import multiprocessing
import traceback
import typing
# third-party imports
import tqdm

# global
CORE_N = max(1, multiprocessing.cpu_count() - 4)

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

    Notes:
    - Process-based execution requires that target functions be
        pickleable.
    - Exceptions in individual jobs are captured and returned as
        dictionaries rather than propagated, unless explicitly re-raised
        for system-level signals.
    '''
    def __init__(
            self,
            max_workers: int = CORE_N,
            use_threads: bool= False,
            show_progress: bool= True,
            *,
            progress_bar_len: int = 100,
            desc: str | None = None
        ):
        '''
        Initialize the executor.

        Args:
            max_workers (int): Number of workers. Default core - 4.
            use_threads (bool): Use threads instead of processes.
            show_progress (bool): Show tqdm progress bar.
            progress_bar_len (int): Width of the progress bar.
            desc (str): Optional leading description for progress bar.
        '''
        self.max_workers = max_workers
        self.use_threads = use_threads
        self.show_progress = show_progress
        self.ncol = progress_bar_len
        self.desc = desc

    def run(self, jobs: list, desc: str | None = None) -> list[typing.Any]:
        '''
        Execute jobs in parallel.

        Args:
            jobs (list): List of tuples (func, args, kwargs).
            desc (str): Optional leading description for progress bar.

        Returns:
            list: Results from executing all jobs.
        '''
        if self.use_threads:
            return self._run_with_threads(jobs, desc=desc)
        return self._run_with_processes(jobs, desc=desc)

    def place_holder(self):
        '''Place holder public method.'''

    def _run_with_threads(self, jobs: list, desc: str | None = None):
        '''Run jobs using ThreadPoolExecutor with error handling.'''
        results = []
        with concurrent.futures.ThreadPoolExecutor(self.max_workers) as exe:
            futures = [
                exe.submit(self._safe_call, func, args, kwargs)
                for func, args, kwargs in jobs
            ]
            it = (f.result() for f in futures)
            if self.show_progress:
                results = list(tqdm.tqdm(it, total=len(jobs), ncols=self.ncol, desc=desc or self.desc))
            else:
                results = list(it)
        return results

    def _run_with_processes(self, jobs: list, desc: str | None = None):
        '''Run jobs using multiprocessing Pool with error handling.'''
        with multiprocessing.Pool(self.max_workers) as pool:
            it = pool.imap(self._wrapper_func, jobs)
            if self.show_progress:
                results = list(tqdm.tqdm(it, total=len(jobs), ncols=self.ncol, desc=desc or self.desc))
            else:
                results = list(it)
        return results

    @staticmethod
    def _wrapper_func(job):
        '''Unpack job tuple for imap with error handling.'''
        func, args, kwargs = job
        return ParallelExecutor._safe_call(func, args, kwargs)

    @staticmethod
    def _safe_call(func, args, kwargs):
        '''Execute a function safely, capturing errors.'''

        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit) as sys_exc:
            raise sys_exc # re-raise syetem-level exceptions
        except Exception as e: # pylint: disable=broad-except
            print('Job failed in', func.__name__, e)
            print({"error": str(e), "traceback": traceback.format_exc()})
            # raise e
            return {"error": str(e), "traceback": traceback.format_exc()}
