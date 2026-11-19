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
Interactive training session process safety and telemetry utilities.

Provides graceful cancellation handling with GPU memory cache cleanup and
MLflow status tagging for notebook sessions.

Public APIs:
    - `safe_training_session`: Interruption-safe training context manager.
'''

# standard imports
import contextlib
import pathlib
import time
import typing
# third-party imports
import mlflow
import mlflow.exceptions
import torch


# ----- public context managers
@contextlib.contextmanager
def safe_training_session(
    target_dpath: str | pathlib.Path | None = None,
    *,
    reraise: bool = False,
    **kwargs: typing.Any,
) -> typing.Generator[dict[str, typing.Any], None, None]:
    '''
    Execute training session with graceful cancellation and process safety.

    Intercepts `KeyboardInterrupt` / notebook cell stop signals to:
    1. Mark active MLflow run status as INTERRUPTED (if active).
    2. Release PyTorch CUDA memory cache to prevent dangling allocations.
    3. Print a clean, actionable status notification.

    Args:
        target_dpath:
            Optional output or experiment directory for compatibility.
        reraise:
            If True, re-raises `KeyboardInterrupt` after cleanup.
            If False (default for notebook interactive sessions), cleanly
            halts cell execution without flooding notebook with tracebacks.
        **kwargs:
            Additional keyword arguments accepted for compatibility.

    Yields:
        dict[str, typing.Any]:
            Mutable context state dictionary recording execution status.
    '''
    _ = target_dpath, kwargs
    session_state: dict[str, typing.Any] = {
        'interrupted': False,
        'start_time': time.time(),
    }

    try:
        yield session_state
    except KeyboardInterrupt as exc:
        session_state['interrupted'] = True

        # tag active MLflow run if present
        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.set_tag('status', 'INTERRUPTED')
                mlflow.set_tag('interrupted_at', time.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.end_run(status='KILLED')
        except (mlflow.exceptions.MlflowException, AttributeError):
            pass

        # release CUDA allocations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # inform user cleanly
        print(
            '\n[INTERRUPTED] Training session safely halted by user.\n'
            '- In-flight GPU memory cache released.\n'
            '- Completed epoch checkpoints preserved in run directory.'
        )

        if reraise:
            raise exc
