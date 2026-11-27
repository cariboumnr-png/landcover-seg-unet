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
Structured checkpoint catalog and final-checkpoint resolution.

Tracks phase-scoped `best` and `last` checkpoint metadata inside
`checkpoints/status.json` (`run.checkpoint_catalog`) and resolves the
canonical final checkpoint for a completed or interrupted session run.

Public APIs:
    - `CheckpointCatalog`: Reader, writer, and resolver for session runs.
    - `PhaseCheckpointEntry`: Per-phase checkpoint paths and metrics.
    - `SCHEMA_ID`: Versioned schema identifier for `status.json`.
'''

# standard imports
import dataclasses
import math
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts

# ----- public constants
SCHEMA_ID = 'checkpoint_catalog/v0.1.0'
_POLICY_LAST_PHASE_BEST = 'last_phase_best'


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class PhaseCheckpointEntry:
    '''
    Checkpoint file paths and metric summary for a single phase.

    Attributes:
        best_fpath:
            Resolved path to the phase best checkpoint, if recorded.
        last_fpath:
            Resolved path to the phase last checkpoint, if recorded.
        best_epoch:
            Epoch index corresponding to `best_fpath`, if recorded.
        best_metric:
            Tracked metric value for `best_fpath`, if finite.
        last_epoch:
            Most recent checkpointed epoch index, if recorded.
        completed:
            Whether the phase reached `PhaseEnd` or `StopRun`.
    '''
    best_fpath: pathlib.Path | None = None
    last_fpath: pathlib.Path | None = None
    best_epoch: int | None = None
    best_metric: float | None = None
    last_epoch: int | None = None
    completed: bool = False


# ----- public classes
class CheckpointCatalog:
    '''
    Structured checkpoint catalog backed by `checkpoints/status.json`.

    Provides side-effect-free lookup and final-checkpoint selection on
    read, plus incremental phase updates during training execution.
    '''

    def __init__(
        self,
        *,
        run: artifacts.SessionPaths,
        phases: dict[str, PhaseCheckpointEntry] | None = None,
        policy: str = _POLICY_LAST_PHASE_BEST,
    ) -> None:
        '''
        Initialize a checkpoint catalog handle bound to `run`.

        Args:
            run:
                Bound `SessionPaths` instance for the target session.
            phases:
                Optional ordered mapping of phase names to entries.
            policy:
                Final checkpoint selection policy name.
        '''
        self._run = run
        self._run_folder = run.effective_run_folder
        self._catalog_fpath = run.checkpoint_catalog
        self._phases: dict[str, PhaseCheckpointEntry] = dict(phases or {})
        self._policy = policy

    @classmethod
    def from_run(cls, run: artifacts.SessionPaths) -> 'CheckpointCatalog':
        '''
        Load checkpoint catalog from `run`, with legacy fallback.

        Reads `run.checkpoint_catalog` when present. When absent on pre-
        catalog runs, accepts exactly one legacy `*_best.pt` file under
        `run.checkpoints`; raises `ArtifactError` when zero or multiple
        legacy candidates exist.

        Args:
            run:
                Bound `SessionPaths` instance for the target session.

        Returns:
            CheckpointCatalog:
                Populated catalog for resolving phase or final weights.

        Raises:
            ArtifactError:
                If `run` is unbound, catalog schema/payload is invalid,
                or legacy fallback finds 0 or >1 `*_best.pt` files.
        '''
        run_folder = run.effective_run_folder
        catalog_fpath = run.checkpoint_catalog
        if catalog_fpath.is_file():
            ctrl = artifacts.Controller[dict[str, typing.Any]](
                catalog_fpath,
                schema_id=SCHEMA_ID,
            )
            payload = ctrl.fetch()
            return cls._from_payload(run=run, payload=payload)

        ckpt_dir = run.checkpoints
        candidates = (
            sorted(ckpt_dir.glob('*_best.pt'))
            if ckpt_dir.is_dir()
            else []
        )
        if len(candidates) == 1:
            best_fp = candidates[0]
            phase_name = _infer_legacy_phase_name(best_fp)
            last_candidate = ckpt_dir / f'{best_fp.stem[:-5]}_last.pt'
            last_fp = last_candidate if last_candidate.is_file() else None
            entry = PhaseCheckpointEntry(best_fpath=best_fp, last_fpath=last_fp)
            return cls(run=run, phases={phase_name: entry})

        if not candidates:
            raise artifacts.ArtifactError(
                f'No checkpoint catalog ({catalog_fpath}) or legacy '
                f'*_best.pt checkpoint found under {run_folder}'
            )
        names = [c.name for c in candidates]
        raise artifacts.ArtifactError(
            f'Ambiguous legacy checkpoints under {ckpt_dir} without '
            f'{catalog_fpath.name}: found {len(candidates)} *_best.pt files '
            f'({names})'
        )

    @property
    def run_folder(self) -> pathlib.Path:
        '''Return the session run directory path.'''
        return self._run_folder

    @property
    def policy(self) -> str:
        '''Return the configured final-checkpoint selection policy.'''
        return self._policy

    @property
    def phases(self) -> dict[str, PhaseCheckpointEntry]:
        '''Return ordered shallow copy of per-phase checkpoint entries.'''
        return dict(self._phases)

    @property
    def final(self) -> pathlib.Path:
        '''
        Resolve the final checkpoint path according to `self.policy`.

        For policy `'last_phase_best'`, selects the `best_fpath` of the
        last phase in curriculum order that recorded one, falling back
        to earlier phases only when later phases never recorded a best
        checkpoint.

        Returns:
            pathlib.Path:
                Existing final checkpoint `.pt` path.

        Raises:
            ArtifactError:
                If policy is unknown, no phase recorded a best
                checkpoint, or the selected file is missing on disk.
        '''
        if self._policy != _POLICY_LAST_PHASE_BEST:
            raise artifacts.ArtifactError(
                f'Unsupported checkpoint selection policy: {self._policy!r}'
            )

        for phase_name, entry in reversed(list(self._phases.items())):
            if entry.best_fpath is None:
                continue
            if not entry.best_fpath.is_file():
                raise artifacts.ArtifactError(
                    f'Recorded best checkpoint for phase "{phase_name}" does '
                    f'not exist: {entry.best_fpath}'
                )
            return entry.best_fpath

        raise artifacts.ArtifactError(
            f'No best checkpoint found in checkpoint catalog: '
            f'{self._catalog_fpath}'
        )

    def best(self, phase: str) -> pathlib.Path:
        '''
        Return the existing best checkpoint path for `phase`.

        Args:
            phase:
                Name of the training phase to look up.

        Returns:
            pathlib.Path:
                Existing `.pt` path for the phase best checkpoint.

        Raises:
            ArtifactError:
                If `phase` is unknown, has no recorded best checkpoint,
                or its file is missing on disk.
        '''
        entry = self._require_phase(phase)
        if entry.best_fpath is None:
            raise artifacts.ArtifactError(
                f'No best checkpoint recorded for phase "{phase}"'
            )
        if not entry.best_fpath.is_file():
            raise artifacts.ArtifactError(
                f'Recorded best checkpoint for phase "{phase}" does not '
                f'exist: {entry.best_fpath}'
            )
        return entry.best_fpath

    def last(self, phase: str) -> pathlib.Path:
        '''
        Return the existing last checkpoint path for `phase`.

        Args:
            phase:
                Name of the training phase to look up.

        Returns:
            pathlib.Path:
                Existing `.pt` path for the phase last checkpoint.

        Raises:
            ArtifactError:
                If `phase` is unknown, has no recorded last checkpoint,
                or its file is missing on disk.
        '''
        entry = self._require_phase(phase)
        if entry.last_fpath is None:
            raise artifacts.ArtifactError(
                f'No last checkpoint recorded for phase "{phase}"'
            )
        if not entry.last_fpath.is_file():
            raise artifacts.ArtifactError(
                f'Recorded last checkpoint for phase "{phase}" does not '
                f'exist: {entry.last_fpath}'
            )
        return entry.last_fpath

    def record_save(
        self,
        phase: str,
        *,
        last_fpath: pathlib.Path,
        epoch: int,
        best_fpath: pathlib.Path | None = None,
        metric: float | None = None,
    ) -> None:
        '''
        Record checkpoint save metadata for `phase` in memory.

        Preserves any previously recorded `best_fpath`, `best_epoch`, and
        `best_metric` when `best_fpath` is `None`.

        Args:
            phase:
                Name of the active training phase.
            last_fpath:
                Path where the latest epoch checkpoint was saved.
            epoch:
                Phase-local epoch index just saved.
            best_fpath:
                Optional path where a new best checkpoint was saved.
            metric:
                Tracked target metric for this epoch (`+/-inf` and `nan`
                are sanitized to `None` when stored).
        '''
        prev = self._phases.get(phase, PhaseCheckpointEntry())
        next_best_fpath = best_fpath if best_fpath is not None else prev.best_fpath
        next_best_epoch = epoch if best_fpath is not None else prev.best_epoch
        next_best_metric = (
            _sanitize_float(metric)
            if best_fpath is not None
            else prev.best_metric
        )
        self._phases[phase] = PhaseCheckpointEntry(
            best_fpath=next_best_fpath,
            last_fpath=last_fpath,
            best_epoch=next_best_epoch,
            best_metric=next_best_metric,
            last_epoch=epoch,
            completed=prev.completed,
        )

    def mark_phase_completed(self, phase: str) -> None:
        '''
        Mark `phase` as completed in the in-memory catalog.

        Args:
            phase:
                Name of the finished training phase.
        '''
        prev = self._phases.get(phase, PhaseCheckpointEntry())
        self._phases[phase] = PhaseCheckpointEntry(
            best_fpath=prev.best_fpath,
            last_fpath=prev.last_fpath,
            best_epoch=prev.best_epoch,
            best_metric=prev.best_metric,
            last_epoch=prev.last_epoch,
            completed=True,
        )

    def persist(self) -> None:
        '''Persist catalog state to `run.checkpoint_catalog` JSON.'''
        payload = self.to_payload()
        ctrl = artifacts.Controller[dict[str, typing.Any]](
            self._catalog_fpath,
            schema_id=SCHEMA_ID,
        )
        ctrl.persist(payload)

    def to_payload(self) -> dict[str, typing.Any]:
        '''
        Serialize catalog into the canonical JSON envelope dictionary.

        Checkpoint paths are stored relative to `self.run_folder` using
        POSIX separators so run folders remain relocatable.

        Returns:
            dict[str, typing.Any]:
                Serializable catalog dictionary including `schema_id`.
        '''
        phases_payload: dict[str, dict[str, typing.Any]] = {}
        for phase_name, entry in self._phases.items():
            phases_payload[phase_name] = {
                'best_fpath': _to_rel_posix(entry.best_fpath, self._run_folder),
                'last_fpath': _to_rel_posix(entry.last_fpath, self._run_folder),
                'best_epoch': entry.best_epoch,
                'best_metric': _sanitize_float(entry.best_metric),
                'last_epoch': entry.last_epoch,
                'completed': entry.completed,
            }
        return {
            'schema_id': SCHEMA_ID,
            'policy': self._policy,
            'phases': phases_payload,
        }

    # ----- private helpers
    def _require_phase(self, phase: str) -> PhaseCheckpointEntry:
        '''Return entry for `phase` or raise `ArtifactError`.'''
        if phase not in self._phases:
            raise artifacts.ArtifactError(
                f'Phase "{phase}" not found in checkpoint catalog '
                f'{self._catalog_fpath}'
            )
        return self._phases[phase]

    @classmethod
    def _from_payload(
        cls,
        *,
        run: artifacts.SessionPaths,
        payload: typing.Any,
    ) -> 'CheckpointCatalog':
        '''Deserialize catalog dictionary loaded via `Controller`.'''
        if not isinstance(payload, dict):
            raise artifacts.ArtifactError(
                f'Invalid checkpoint catalog payload in {run.checkpoint_catalog}'
            )
        policy = payload.get('policy', _POLICY_LAST_PHASE_BEST)
        phases_raw = payload.get('phases')
        if not isinstance(policy, str) or not isinstance(phases_raw, dict):
            raise artifacts.ArtifactError(
                f'Invalid checkpoint catalog fields in {run.checkpoint_catalog}'
            )
        run_folder = run.effective_run_folder
        phases: dict[str, PhaseCheckpointEntry] = {}
        for phase_name, raw_entry in phases_raw.items():
            if not isinstance(phase_name, str) or not isinstance(raw_entry, dict):
                raise artifacts.ArtifactError(
                    f'Invalid phase entry in {run.checkpoint_catalog}'
                )
            phases[phase_name] = PhaseCheckpointEntry(
                best_fpath=_from_rel_posix(
                    raw_entry.get('best_fpath'), run_folder
                ),
                last_fpath=_from_rel_posix(
                    raw_entry.get('last_fpath'), run_folder
                ),
                best_epoch=raw_entry.get('best_epoch'),
                best_metric=_sanitize_float(raw_entry.get('best_metric')),
                last_epoch=raw_entry.get('last_epoch'),
                completed=bool(raw_entry.get('completed', False)),
            )
        return cls(run=run, phases=phases, policy=policy)


# ----- private helpers
def _sanitize_float(value: typing.Any) -> float | None:
    '''Convert non-finite numeric values (`+/-inf`, `nan`) to `None`.'''
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _to_rel_posix(
    fpath: pathlib.Path | None,
    run_folder: pathlib.Path,
) -> str | None:
    '''Convert `fpath` inside `run_folder` to a relative POSIX string.'''
    if fpath is None:
        return None
    try:
        return fpath.relative_to(run_folder).as_posix()
    except ValueError:
        try:
            return fpath.resolve().relative_to(run_folder.resolve()).as_posix()
        except ValueError:
            return fpath.as_posix()


def _from_rel_posix(
    raw: typing.Any,
    run_folder: pathlib.Path,
) -> pathlib.Path | None:
    '''Resolve relative POSIX checkpoint path string under `run_folder`.'''
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise artifacts.ArtifactError(f'Invalid checkpoint path entry: {raw!r}')
    candidate = pathlib.PurePosixPath(raw)
    if candidate.is_absolute():
        return pathlib.Path(raw)
    resolved_root = run_folder.resolve()
    resolved_target = (run_folder / candidate).resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise artifacts.ArtifactError(
            f'Checkpoint path escapes run folder: {raw!r}'
        ) from exc
    return run_folder / candidate


def _infer_legacy_phase_name(best_fp: pathlib.Path) -> str:
    '''Derive a display phase name from a legacy `*_best.pt` filename.'''
    stem = best_fp.stem
    if stem.endswith('_epoch_best'):
        return stem[:-11] or 'default'
    if stem.endswith('_best'):
        return stem[:-5] or 'default'
    return stem or 'default'
