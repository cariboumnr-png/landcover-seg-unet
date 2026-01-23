'''
A module contains utility functions used throughout this project.

Functions:
    check_valid_csv(): Checks if a csv file exists and contains data.
    check_gpu(): Checks GPU availability for parallel computing.
    get_file_ctime(): Get file creation time as a formatted string.
    get_timestamp(): Get current time as a formatted string.
    timed_tun(): Runs a function with a timer.
    ...
'''

# standard imports
import csv
import datetime
import inspect
import json
import os
import pickle
import subprocess
import sys
import typing
# third-party imports
import torch
# local imports
import utils.logger

def check_valid_csv(csv_path: str) -> bool:
    '''Checks if a csv file exists and contains data.'''

    # FALSE if file does not exist
    if not os.path.exists(csv_path):
        return False

    # FALSE if only contains a header (no content from row 1)
    with open(csv_path, mode='r', encoding='utf-8') as file:
        rows = list(csv.reader(file))
        if not rows[1:]:
            return False

    # TRUE otherwise
    return True

def check_gpu(log: utils.logger.Logger) -> torch.device:
    '''
    Checks GPU availability for parallel computing.

    This function detects whether a GPU is available, and if so, logs
    information inclduing:
    * GPU model
    * CUDA version
    * Number of CUDA cores
    * Memory usage (total, allocated, and free memory)
    * GPU driver version
    * GPU utilization
    * GPU temperature
    * Active GPU processes (if any)

    If any of the optional checks fail, errors are logged while the
    function continues executing without interruption.

    Args:
        log (app.utils_logger.Logger): Logger for logging GPU information.

    Returns:
        str: `'cuda'` if GPU is available, else `'cpu'`

    Example:
        >>> check_gpu()
        'cuda'
        # example logs
        ...log-DEBUG - Retrieving GPU information...
        ...log-DEBUG -  | GPU available: NVIDIA T600 Laptop GPU
        ...log-DEBUG -  | CUDA ver.: 11.8
        ...log-DEBUG -  | Number of CUDA cores: 256
        ...log-DEBUG -  | Total/Allocated/Free Memory: ...GB/...GB/...GB
        ...log-DEBUG -  | Driver ver.: 538.78
        ...log-DEBUG -  | GPU util.: 0%
        ...log-DEBUG -  | GPU Temp.: 53 C
        ...log-DEBUG -  | Error retrieving GPU processes: ...

    --------------------------------------------------------------------
    Notes: The function logs useful information for debugging in case of
    errors (e.g., issues retrieving GPU processes or utilization stats).
    This is useful for monitoring the GPU status in an environment where
    GPU resources are crucial for parallel computation.
    '''

    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')

        # get GPU info
        gpu_name = torch.cuda.get_device_name(device)
        cuda_ver = torch.version.cuda  # type: ignore
        num_cuda_cores = torch.cuda.device_count() * 256  # Approximation
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory

        # log general info
        log.log('INFO', f' |  - GPU available: {gpu_name}')
        log.log('INFO', f' |  - CUDA ver.: {cuda_ver}')
        log.log('INFO', f' |  - Number of CUDA cores: {num_cuda_cores}')
        log.log('INFO', f' |  - Total/Allocated/Free Memory: '
                            f'{total_memory / (1024 ** 3):.2f} GB/'
                            f'{allocated_memory / (1024 ** 3):.2f} GB/'
                            f'{free_memory / (1024 ** 3):.2f} GB')

        # GPU driver version
        try:
            data = subprocess.check_output([
                'nvidia-smi', '--query-gpu=driver_version',
                '--format=csv,noheader,nounits'
            ])
            log.log('DEBUG', f' |  - Driver ver.: '
                                f'{data.decode("utf-8").strip()}')
        except subprocess.CalledProcessError as e:
            log.log('DEBUG', f' |  - Error retrieving GPU driver ver: {e}')

        # GPU utilization
        try:
            data = subprocess.check_output([
                'nvidia-smi', '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ])
            log.log('DEBUG', f' |  - GPU util.: '
                                f'{data.decode("utf-8").strip()}%')
        except subprocess.CalledProcessError as e:
            log.log('DEBUG', f' |  - Error retrieving GPU util.: {e}')

        # GPU temperature
        try:
            data = subprocess.check_output([
                'nvidia-smi', '--query-gpu=temperature.gpu',
                '--format=csv,noheader,nounits'
            ])
            log.log('DEBUG', f' |  - GPU temp.: '
                                f'{data.decode("utf-8").strip()} C')
        except subprocess.CalledProcessError as e:
            log.log('DEBUG', f' |  - Error retrieving GPU temp.: {e}')

        # GPU processes
        try:
            data = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid,process_name,'
                'gpu_memory_usage', '--format=csv,noheader'
            ])
            log.log('DEBUG', f' |  - GPU Processes:'
                                f'\n{data.decode("utf-8")}')
        except subprocess.CalledProcessError as e:
            log.log('DEBUG', f' |  - Error retrieving GPU processes: {e}')

        # return
        return device

    # if no GPU is available, fall back to CPU
    log.log('INFO', ' |  - No GPU found - CPU will be used instead.')
    return torch.device('cpu')

def get_file_ctime(filepath: str, t_format: str='%Y%m%d_%H%M%S') -> str:
    '''
    Get file creation time as a string specified by `t_format`.

    Args:
        filepath (str): To the file to be checked.
        t_format (str, optional): Sets time string format
            (default: 20001234_567).
    '''

    # get creation time
    creation_time = os.path.getctime(filepath)
    # format and return
    return datetime.datetime.fromtimestamp(creation_time).strftime(t_format)

def get_timestamp(t_format: str='%Y%m%d_%H%M%S') -> str:
    '''
    Get current time as a string specified by `t_format`.

    Args:
        t_format (str, optional): Sets time string format
            (default: 20001234_567).
    '''

    # return formatted time string
    return datetime.datetime.now().strftime(t_format)

def timed_run(func, log: utils.logger.Logger, args: object | None=None,
              kwargs: dict | None=None, n: int | None=None) -> object:
    '''
    Runs a function with a timer.

    This function uses a customized Logger class for logging, where it
    tracks the total execution time of a function. If an loop count `n`
    argument is provided, it logs the processing speed in items per
    second (`it/s`) or seconds per item (`s/it`).

    Callable function with the following signature are supported:
    * func() # no arguments
    * func(a) # single element argument
    * func(*args) # only positional arguments
    * func(**kwargs) # only keyword arguments
    * func(*args, **kwargs) # both positional and keyword arguments

    Args:
        func (Callable): The function to be timed.
        log (app.utils_logger.Logger): Logs information.
        args (any, optional): Positional arguments for `func`
            (default: None).
        kwargs (dict, optional): Keyword arguments for `func`
            (default: None).
        n (int, optional): An internal count of iterables in the
            function, used to log processing speed (default: None).

    Returns:
        any: The return value of the `func` being called. `None` if the
            function does not have a return value.

    Raises:
        AttributeError: e.g., `func` is not a valid attribute.
        RuntimeError: e.g., runtime error occurs during `func` calling.
        TypeError: e.g., wrong argument number of type passed to `func`.
        ValueError: e.g., inappropriate or out-range arguments

    Example:
    >>> # Example function to be timed
    >>> def eg_func(data):
    >>>    return sum(data)
    >>> # Example usage
    >>> result = timed_run(eg_func, log=my_logger, args=[1, 2, 3, 4])
    '''

    # normalize input args
    if args is None:
        args = ()
    elif not isinstance(args, (tuple, list)): # allow list to be unpacked too
        args = (args,)
    # normalize input kwargs
    if kwargs is None:
        kwargs = {}
    elif kwargs is not None and not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary if provided.")

    # start logging with module and function name
    log.log_sep()
    t0 = datetime.datetime.now() # timer on
    log.log('INFO', f'[Started] {func.__name__}()@{func.__module__}.py')

    # get signature of func
    sig = inspect.signature(func)
    # set an empty returning result
    result = None
    # use try-except block to run the input function
    try:
        # try match signature and provided arguments
        sig.bind(*args, **kwargs) # raise TypeError if can't bind
        # call func if binding is successful
        result = func(*args, **kwargs)
    # capture exceptions during function call, e.g. improper arguments
    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
        log.log('ERROR', f'{e}')
        raise

    # log total time
    tt = datetime.datetime.now() - t0 # timer off
    log.log('INFO', f'[Finished] {func.__name__}()@{func.__module__}.py')
    log.log('INFO', f'Elapsed time: {tt}')

    # log processing speed if n is provided
    if n:
        if n > tt.total_seconds():
            sp = round(n / tt.total_seconds(), 2)
            log.log('INFO', f'{n} items processed at average speed: {sp} it/s')
        else:
            sp = round(tt.total_seconds() / n, 2)
            log.log('INFO', f'{n} items processed at average speed: {sp} s/it')

    # return the function outcome
    log.log_sep()
    return result

def print_status(lines: list):
    '''Helper to print multiple lines refreshing'''

    print('\n')
    # Calculate the number of lines that should be refreshed
    num_lines_to_clear = len(lines)
    # Move the cursor up by that number of lines
    sys.stdout.write(f'\033[{num_lines_to_clear}F')
    # Move cursor up by len(lines) and clear lines using ANSI escape codes
    sys.stdout.write('\033[F' * len(lines))  # Move cursor up
    for line in lines:
        sys.stdout.write('\033[K')  # Clear the line
        print(line)
    print('\n')

def load_json(json_fpath: str) -> typing.Any:
    '''Helper to load a json config file.'''

    with open(json_fpath, 'r', encoding='UTF-8') as src:
        return json.load(src)

def write_json(json_fpath: str, src_dict: list | dict) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    with open(json_fpath, 'w', encoding='UTF-8') as file:
        json.dump(src_dict, file, indent=4)

def load_pickle(pickle_fpath: str) -> typing.Any:
    '''Helper to load a .pickle file'''

    with open(pickle_fpath, 'rb') as file:
        return pickle.load(file)

def write_pickle(pickle_fpath: str, src_obj: typing.Any) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    with open(pickle_fpath, 'wb') as file:
        pickle.dump(src_obj, file)

def get_fpaths_from_dir(dirpath: str, suffix: str | None=None) -> list:
    '''List all files from a directory with optional suffix filter.'''

    fpaths = []
    for root, _, files in os.walk(dirpath):
        for file in files:
            if not suffix:
                fpaths.append(os.path.join(root, file))
            else:
                if file.endswith(suffix):
                    fpaths.append(os.path.join(root, file))
    return fpaths
