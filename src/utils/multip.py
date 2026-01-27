'''Simple multiprocessing framework wrapped in a class.'''

# standard imports
import concurrent.futures
import multiprocessing
import traceback
import typing
# third-party imports
import tqdm

# global
CORE_N = multiprocessing.cpu_count() - 4

class ParallelExecutor:
    '''Doc.'''
    def __init__(
            self,
            max_workers=CORE_N,
            use_threads=False,
            show_progress=True
        ):
        '''
        Initialize the executor.

        Args:
            max_workers (int): Number of workers. Default core - 4.
            use_threads (bool): Use threads instead of processes.
            show_progress (bool): Show tqdm progress bar.
        '''
        self.max_workers = max_workers
        self.use_threads = use_threads
        self.show_progress = show_progress

    def run(self, jobs: list) -> list[typing.Any]:
        '''
        Execute jobs in parallel.

        Args:
            jobs (list): List of tuples (func, args, kwargs).

        Returns:
            list: Results from executing all jobs.
        '''
        if self.use_threads:
            return self._run_with_threads(jobs)
        return self._run_with_processes(jobs)

    def place_holder(self):
        '''Place holder public method.'''

    def _run_with_threads(self, jobs: list):
        '''Run jobs using ThreadPoolExecutor with error handling.'''
        results = []
        with concurrent.futures.ThreadPoolExecutor(self.max_workers) as exe:
            futures = [
                exe.submit(self._safe_call, func, args, kwargs)
                for func, args, kwargs in jobs
            ]
            iterator = (f.result() for f in futures)
            if self.show_progress:
                results = list(tqdm.tqdm(iterator, total=len(jobs)))
            else:
                results = list(iterator)
        return results

    def _run_with_processes(self, jobs: list):
        '''Run jobs using multiprocessing Pool with error handling.'''
        with multiprocessing.Pool(self.max_workers) as pool:
            iterator = pool.imap(self._wrapper_func, jobs)
            if self.show_progress:
                results = list(tqdm.tqdm(iterator, total=len(jobs)))
            else:
                results = list(iterator)
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
