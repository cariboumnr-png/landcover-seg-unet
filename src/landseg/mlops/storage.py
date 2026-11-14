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
MLOps storage, experiment path, and cloud volume resolution utilities.

Provides precedence-based resolution of experiment root workspaces across
cloud volumes (e.g. Unity Catalog Volumes `/Volumes/...`), local POSIX paths,
and environment variables. Also manages environment-safe Optuna storage
connection strings and synchronization between driver scratch and persistent volumes.

Public APIs:
    - `load_study_trials`: Load persisted sweep trials dataframe or summary.
    - `resolve_exp_root`: Resolve experiment root path across contexts.
    - `resolve_optuna_storage`: Resolve environment-safe Optuna URI.
    - `sync_optuna_study`: Persist Optuna study trials to volume storage.
    - `validate_output_root`: Validate user-designated session output root.
'''

# standard imports
import json
import os
import pathlib
import shutil
import typing
# third-party imports
import optuna
import pandas
# local imports
import landseg.mlops.env as env_mod
import landseg.study.analysis.trials as trials_analysis


# ----- public functions
def resolve_exp_root(
    exp_root: str | pathlib.Path | None = None,
    default: str | pathlib.Path = './experiment',
    *,
    env_var: str = 'LANDSEG_EXP_ROOT',
    create_if_missing: bool = False,
    dbutils: typing.Any = None,
) -> pathlib.Path:
    '''
    Resolve experiment root path across local, CI, and Databricks.

    Evaluation precedence:
    1. Explicit `exp_root` argument if provided and non-empty.
    2. Databricks widget or dbutils parameter if available (`output_root` or `exp_root`).
    3. Dedicated environment variable (default: `LANDSEG_OUTPUT_ROOT`, `LANDSEG_EXP_ROOT`,
       fallback: `EXP_ROOT`).
    4. Fallback default (`./experiment`).

    Args:
        exp_root:
            Optional explicit directory path.
        default:
            Fallback directory if no other source is configured.
        env_var:
            Environment variable name to check.
        create_if_missing:
            If True, automatically create the directory hierarchy.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        pathlib.Path:
            Resolved experiment root directory.
    '''
    if exp_root is not None and str(exp_root).strip():
        target = pathlib.Path(str(exp_root).strip())
    else:
        db = env_mod.get_dbutils(dbutils)
        widget_val: str | None = None
        if db is not None and hasattr(db, 'widgets'):
            for wname in ('output_root', 'exp_root'):
                try:
                    wval = db.widgets.get(wname)
                    if wval and str(wval).strip():
                        widget_val = str(wval).strip()
                        break
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        if widget_val:
            target = pathlib.Path(widget_val)
        else:
            env_val = (
                os.environ.get('LANDSEG_OUTPUT_ROOT') or
                os.environ.get(env_var) or
                os.environ.get('EXP_ROOT')
            )
            if env_val and env_val.strip():
                target = pathlib.Path(env_val.strip())
            else:
                target = pathlib.Path(default)

    target = target.expanduser()
    if create_if_missing:
        target.mkdir(parents=True, exist_ok=True)
    return target


def validate_output_root(
    path: str | pathlib.Path | None,
    *,
    probe_write: bool = True,
    dbutils: typing.Any = None,
) -> pathlib.Path:
    '''
    Validate a user-designated session output root directory.

    Enforces cloud filesystem safety policies:
    1. Rejects empty or missing output paths (with actionable guidance).
    2. Explicitly blocks Databricks `/Workspace/...` paths because workspace files
       cannot store binary model checkpoints (.pt) due to Databricks flush limits.
    3. On Databricks clusters, enforces Unity Catalog Volume paths (`/Volumes/...`)
       or cluster scratch (`/local_disk0`).
    4. Probes directory writability (via test probe file creation and removal)
       before heavy computation is committed.

    Args:
        path:
            Output directory string or pathlib.Path.
        probe_write:
            If True, verifies filesystem writability by writing and deleting
            a sentinel probe file.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        pathlib.Path:
            Validated output directory path.

    Raises:
        ValueError:
            If path is empty, invalid format, or points to disallowed storage.
        PermissionError:
            If target directory is not writable.
    '''
    _ = dbutils
    raw_str = ''
    if path is not None:
        raw_str = str(path).strip()

    if not raw_str:
        # Check env var fallback for headless CI parity
        env_val = (
            os.environ.get('LANDSEG_OUTPUT_ROOT') or
            os.environ.get('LANDSEG_EXP_ROOT')
        )
        if env_val and env_val.strip():
            raw_str = env_val.strip()

    if not raw_str:
        raise ValueError(
            "Output root path is empty. Please set the 'output_root' widget "
            "(or 'LANDSEG_OUTPUT_ROOT' environment variable) to a writable "
            "Unity Catalog Volume path (e.g. '/Volumes/<catalog>/<schema>/<volume>/<dir>') "
            "or local directory before proceeding."
        )

    norm_path = raw_str.replace('\\', '/')

    # Guard 1: Reject /Workspace paths
    if norm_path.startswith('/Workspace') or norm_path.startswith('Workspace/'):
        raise ValueError(
            f"Invalid output location '{raw_str}': Databricks Workspace files "
            "(/Workspace/...) cannot store model checkpoint binaries (.pt) due to "
            "Databricks workspace filesystem flush restrictions. "
            "Please use a Unity Catalog Volume (/Volumes/<catalog>/<schema>/<volume>/<dir>) "
            "or local cluster scratch (/local_disk0)."
        )

    # Guard 2: Enforce UC Volumes or local scratch when running on Databricks
    if env_mod.is_databricks():
        valid_prefixes = ('/Volumes/', '/local_disk0', '/tmp')
        if not any(norm_path.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid output location '{raw_str}' for Databricks runtime. "
                "Session outputs must be written to a Unity Catalog Volume "
                "('/Volumes/<catalog>/<schema>/<volume>/...') or cluster scratch "
                "('/local_disk0')."
            )

    target = pathlib.Path(raw_str).expanduser()

    # Guard 3: Writability probe
    if probe_write:
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe_file = target / '.landseg_probe'
            probe_file.write_text('probe', encoding='utf-8')
            probe_file.unlink(missing_ok=True)
        except Exception as err:
            raise PermissionError(
                f"Output directory '{target}' is not writable: {err}. "
                "Please verify volume existence and write permissions."
            ) from err

    return target


def resolve_optuna_storage(
    storage: str | None = None,
    default: str | None = None,
    *,
    widget_name: str = 'optuna_storage',
    dbutils: typing.Any = None,
) -> str:
    '''
    Resolve safe Optuna storage backend URI across runtimes.

    SQLite on shared cloud storage (DBFS or Unity Catalog Volumes)
    lacks robust POSIX file locking and can corrupt trial databases.
    When running on Databricks without an explicit remote RDBMS URI,
    this helper automatically routes SQLite to driver storage (/tmp/optuna.db).

    Args:
        storage:
            Optional explicit storage connection string.
        default:
            Optional custom default storage string.
        widget_name:
            Widget parameter name to inspect when in Databricks.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        str:
            Environment-safe Optuna storage connection URI.
    '''
    if storage is not None:
        return storage

    db = env_mod.get_dbutils(dbutils)
    if default is not None:
        fallback = default
    elif env_mod.is_databricks():
        fallback = 'sqlite:////tmp/optuna.db'
    else:
        fallback = 'sqlite:///optuna.db'

    if db is not None and hasattr(db, 'widgets'):
        try:
            widget_val = db.widgets.get(widget_name)
            if widget_val:
                return widget_val
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    env_storage = os.environ.get('LANDSEG_OPTUNA_STORAGE')
    if env_storage:
        return env_storage

    return fallback


def sync_optuna_study(
    study_name: str,
    exp_root: str | pathlib.Path | None = None,
    *,
    storage: str | None = None,
    top_k: int = 5,
    dbutils: typing.Any = None,
) -> dict[str, typing.Any]:
    '''
    Synchronize Optuna sweep study summaries and checkpoints to persistent storage.

    When running on Databricks, Optuna SQLite trial databases reside on
    driver-local scratch storage (/tmp/optuna.db) to avoid distributed FUSE
    POSIX locking failures on UC Volumes. This utility persists the final study
    trials dataframe, summary metrics, and Pareto-optimal trial checkpoints
    to the persistent experiment root volume tier.

    Args:
        study_name:
            Identifier of the Optuna study to synchronize.
        exp_root:
            Optional persistent experiment root directory.
        storage:
            Optional Optuna storage connection URI.
        top_k:
            Number of top trials to rank in summary.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        dict[str, typing.Any]:
            Summary dictionary recording persisted artifact paths and metrics.
    '''
    resolved_root = resolve_exp_root(exp_root, dbutils=dbutils)
    resolved_storage = resolve_optuna_storage(storage, dbutils=dbutils)

    # Load Optuna study from scratch or remote storage
    study = optuna.load_study(study_name=study_name, storage=resolved_storage)

    # Prepare persistent volume destination
    sweeps_dir = resolved_root / 'artifacts' / 'sweeps' / study_name
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = resolved_root / 'artifacts' / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rank top completed trials
    ranked = trials_analysis.rank_trials(
        study_name=study_name,
        storage=resolved_storage,
        top_k=top_k,
        ascending=False,
    )

    # 2. Export trial summary metadata JSON
    best_value = None
    best_params = {}
    best_trial_num = None
    try:
        best_value = float(study.best_value)
        best_params = dict(study.best_params)
        best_trial_num = int(study.best_trial.number)
    except ValueError:
        pass

    summary_data = {
        'study_name': study_name,
        'storage': resolved_storage,
        'total_trials': len(study.trials),
        'best_trial_number': best_trial_num,
        'best_value': best_value,
        'best_params': best_params,
        'top_trials': ranked,
    }

    summary_file = sweeps_dir / f'{study_name}_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    # 3. Export complete trials dataframe (CSV / Parquet)
    trials_csv = sweeps_dir / f'{study_name}_trials.csv'
    try:
        df = study.trials_dataframe()
        df.to_csv(trials_csv, index=False)
    except Exception:  # pylint: disable=broad-exception-caught
        trials_csv = None

    # 4. Search and sync best trial checkpoint to persistent volume
    best_ckpt_dest = None
    if best_trial_num is not None:
        candidate_ckpts = list(
            (resolved_root / 'artifacts').glob(f'**/*trial_{best_trial_num:04d}*.pt')
        )
        if candidate_ckpts:
            src_ckpt = candidate_ckpts[0]
            dest_ckpt = checkpoints_dir / f'{study_name}_best.pt'
            try:
                shutil.copy2(src_ckpt, dest_ckpt)
                best_ckpt_dest = str(dest_ckpt)
            except OSError:
                best_ckpt_dest = str(src_ckpt)

    print(
        f'[SWEEP SYNC] Optuna study {study_name!r} successfully synchronized:\n'
        f'- Total trials: {len(study.trials)} | Best value: {best_value}\n'
        f'- Summary persisted: {summary_file}\n'
        f'- Trials table: {trials_csv}'
    )
    if best_ckpt_dest:
        print(f'- Best trial checkpoint: {best_ckpt_dest}')

    return {
        'study_name': study_name,
        'storage': resolved_storage,
        'total_trials': len(study.trials),
        'best_value': best_value,
        'best_params': best_params,
        'summary_file': str(summary_file),
        'trials_csv': str(trials_csv) if trials_csv else None,
        'best_checkpoint': best_ckpt_dest,
    }


def load_study_trials(
    exp_root: str | pathlib.Path | None = None,
    study_name: str = 'study_base_test',
    *,
    dbutils: typing.Any = None,
) -> typing.Any:
    '''
    Load persisted study trials dataframe or summary from experiment volume.

    Args:
        exp_root:
            Optional experiment root directory.
        study_name:
            Identifier of the study to load.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        pandas.DataFrame | dict[str, typing.Any]:
            Loaded trials table or parsed summary dictionary.
    '''
    resolved_root = resolve_exp_root(exp_root, dbutils=dbutils)
    sweeps_dir = resolved_root / 'artifacts' / 'sweeps' / study_name
    csv_path = sweeps_dir / f'{study_name}_trials.csv'
    json_path = sweeps_dir / f'{study_name}_summary.json'

    if csv_path.is_file():
        return pandas.read_csv(csv_path)

    if json_path.is_file():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    raise FileNotFoundError(
        f'No persisted study records found for {study_name!r} under {sweeps_dir}'
    )
