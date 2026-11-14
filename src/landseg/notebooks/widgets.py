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
Interactive parameter widget abstraction for notebook environments.

Provides multi-environment parameter widgets bridging Databricks
`dbutils.widgets` and local or headless execution environments.

Public APIs:
    - `NotebookWidgets`: Ergonomic widget wrapper across runtimes.
'''

# standard imports
import os
import pathlib
import typing
# local imports
import landseg.mlops.env as mlops_env


# ----- internal session cache
_SESSION_VALUES: dict[str, typing.Any] = {}


# ----- public exceptions
class MissingRequiredInputError(ValueError):
    '''Raised when a required notebook widget input is missing or invalid.'''


# ----- public classes
class NotebookWidgets:
    '''
    Ergonomic configuration widget abstraction for notebooks.

    Bridges Databricks `dbutils.widgets` when running in Databricks
    runtime and falls back to environment variables and in-memory
    defaults in standard Jupyter or headless environments.
    '''

    def __init__(self, dbutils: typing.Any = None):
        '''Initialize widget bridge with optional explicit dbutils.'''
        self._dbutils = mlops_env.get_dbutils(dbutils)
        self._values: dict[str, typing.Any] = {}

    def _set_val(self, name: str, val: typing.Any) -> typing.Any:
        self._values[name] = val
        _SESSION_VALUES[name] = val
        return val

    @property
    def has_dbutils(self) -> bool:
        '''Return whether active dbutils widget subsystem is available.'''
        return self._dbutils is not None and hasattr(self._dbutils, 'widgets')

    def text(
        self,
        name: str,
        default: str = '',
        label: str | None = None,
    ) -> str:
        '''
        Define and read a text parameter widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default string value when widget is unassigned.
            label:
                Optional human-readable UI label.

        Returns:
            str:
                Current widget value or fallback.
        '''
        display_label = label if label is not None else name
        if self.has_dbutils:
            try:
                self._dbutils.widgets.text(name, default, display_label)
                val = self._dbutils.widgets.get(name)
                return self._set_val(name, val)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass

        env_val = (
            os.environ.get(f'LANDSEG_{name.upper()}') or
            os.environ.get(name.upper())
        )
        val = env_val if env_val is not None else default
        return self._set_val(name, val)

    def dropdown(
        self,
        name: str,
        default: str,
        choices: list[str],
        label: str | None = None,
    ) -> str:
        '''
        Define and read a dropdown choice widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default choice string.
            choices:
                List of selectable options.
            label:
                Optional human-readable UI label.

        Returns:
            str:
                Selected choice value.
        '''
        display_label = label if label is not None else name
        if self.has_dbutils:
            try:
                self._dbutils.widgets.dropdown(
                    name,
                    default,
                    choices,
                    display_label,
                )
                val = self._dbutils.widgets.get(name)
                return self._set_val(name, val)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass

        env_val = (
            os.environ.get(f'LANDSEG_{name.upper()}') or
            os.environ.get(name.upper())
        )
        val = env_val if env_val in choices else default
        return self._set_val(name, val)

    def combobox(
        self,
        name: str,
        default: str,
        choices: list[str],
        label: str | None = None,
    ) -> str:
        '''
        Define and read a combobox choice widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default choice string.
            choices:
                List of selectable suggestions.
            label:
                Optional human-readable UI label.

        Returns:
            str:
                Selected or entered text value.
        '''
        display_label = label if label is not None else name
        if self.has_dbutils:
            try:
                self._dbutils.widgets.combobox(
                    name,
                    default,
                    choices,
                    display_label,
                )
                val = self._dbutils.widgets.get(name)
                return self._set_val(name, val)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass

        env_val = (
            os.environ.get(f'LANDSEG_{name.upper()}') or
            os.environ.get(name.upper())
        )
        val = env_val if env_val is not None else default
        return self._set_val(name, val)

    def multiselect(
        self,
        name: str,
        default: str,
        choices: list[str],
        label: str | None = None,
    ) -> list[str]:
        '''
        Define and read a multi-selection widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default comma-delimited choice string.
            choices:
                List of selectable options.
            label:
                Optional human-readable UI label.

        Returns:
            list[str]:
                List of selected choice strings.
        '''
        display_label = label if label is not None else name
        raw_val = default
        if self.has_dbutils:
            try:
                self._dbutils.widgets.multiselect(
                    name,
                    default,
                    choices,
                    display_label,
                )
                raw_val = self._dbutils.widgets.get(name)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass
        else:
            env_val = (
                os.environ.get(f'LANDSEG_{name.upper()}') or
                os.environ.get(name.upper())
            )
            if env_val is not None:
                raw_val = env_val

        parsed = [
            item.strip() for item in raw_val.split(',')
            if item.strip()
        ]
        return self._set_val(name, parsed)

    def int(
        self,
        name: str,
        default: int,
        label: str | None = None,
    ) -> int:
        '''
        Define and read an integer parameter widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default integer value.
            label:
                Optional human-readable UI label.

        Returns:
            int:
                Parsed integer parameter.
        '''
        raw = self.text(name, default=str(default), label=label)
        try:
            val = int(raw.strip())
        except (ValueError, AttributeError):
            val = default
        return self._set_val(name, val)

    def float(
        self,
        name: str,
        default: float,
        label: str | None = None,
    ) -> float:
        '''
        Define and read a floating point parameter widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default float value.
            label:
                Optional human-readable UI label.

        Returns:
            float:
                Parsed float parameter.
        '''
        raw = self.text(name, default=str(default), label=label)
        try:
            val = float(raw.strip())
        except (ValueError, AttributeError):
            val = default
        return self._set_val(name, val)

    def boolean(
        self,
        name: str,
        default: bool,
        label: str | None = None,
    ) -> bool:
        '''
        Define and read a boolean flag parameter widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default boolean state.
            label:
                Optional human-readable UI label.

        Returns:
            bool:
                Parsed boolean state.
        '''
        str_default = 'true' if default else 'false'
        raw = self.dropdown(
            name=name,
            default=str_default,
            choices=['true', 'false'],
            label=label,
        )
        val = str(raw).strip().lower() in {'true', '1', 'yes', 'y'}
        return self._set_val(name, val)

    def path(
        self,
        name: str,
        default: str | pathlib.Path,
        label: str | None = None,
        *,
        create_if_missing: bool = False,
    ) -> pathlib.Path:
        '''
        Define and read a filesystem or volume path parameter widget.

        Args:
            name:
                Identifier key for the widget.
            default:
                Default path string or pathlib.Path.
            label:
                Optional human-readable UI label.
            create_if_missing:
                If True, ensures target directory exists on disk.

        Returns:
            pathlib.Path:
                Resolved filesystem path.
        '''
        raw = self.text(name, default=str(default), label=label)
        resolved = pathlib.Path(raw).expanduser()
        if create_if_missing:
            resolved.mkdir(parents=True, exist_ok=True)
        return self._set_val(name, resolved)

    def get(self, name: str, default: typing.Any = None) -> typing.Any:
        '''
        Retrieve cached or active widget value.

        Args:
            name:
                Identifier key of the widget.
            default:
                Fallback value if widget is unregistered.

        Returns:
            typing.Any:
                Stored widget value.
        '''
        if name in self._values:
            return self._values[name]
        if name in _SESSION_VALUES:
            return _SESSION_VALUES[name]
        if self.has_dbutils:
            try:
                val = self._dbutils.widgets.get(name)
                self._set_val(name, val)
                return val
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass
        env_val = (
            os.environ.get(f'LANDSEG_{name.upper()}') or
            os.environ.get(name.upper())
        )
        if env_val is not None:
            return env_val
        return default

    def remove(self, name: str) -> None:
        '''
        Remove a widget by name.

        Args:
            name:
                Identifier key of widget to remove.
        '''
        self._values.pop(name, None)
        _SESSION_VALUES.pop(name, None)
        if self.has_dbutils:
            try:
                self._dbutils.widgets.remove(name)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass

    def remove_all(self) -> None:
        '''Remove all registered widgets from active notebook session.'''
        self._values.clear()
        _SESSION_VALUES.clear()
        if self.has_dbutils:
            try:
                self._dbutils.widgets.removeAll()
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass

    def to_dict(self) -> dict[str, typing.Any]:
        '''
        Export current widget values as a dictionary.

        Returns:
            dict[str, typing.Any]:
                Mapping of widget names to resolved values.
        '''
        return dict(self._values)


# ----- public validation functions
def require_widget(
    name: str,
    *,
    validator: typing.Callable[[typing.Any], typing.Any] | None = None,
    error_message: str | None = None,
    default: typing.Any = None,
    widgets: NotebookWidgets | None = None,
    dbutils: typing.Any = None,
) -> typing.Any:
    '''
    Retrieve and validate a required notebook widget value.

    Raises `MissingRequiredInputError` when the widget value is empty,
    missing, or fails validation. When executing inside Databricks, raising this
    exception cleanly halts notebook 'Run All' cell execution before compute
    is committed.

    Args:
        name:
            Widget parameter key name.
        validator:
            Optional callable that validates the retrieved value. If validator
            raises an exception (e.g. ValueError) or returns False, validation fails.
            If validator returns a transformed value (e.g. resolved Path), that
            value is returned.
        error_message:
            Custom actionable error message to raise on failure.
        default:
            Optional fallback value if widget is unset.
        widgets:
            Optional NotebookWidgets instance to query.
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        typing.Any:
            Validated widget value (or transformed value from validator).

    Raises:
        MissingRequiredInputError:
            If widget is missing, empty, or fails validator.
    '''
    val = None
    if widgets is not None:
        val = widgets.get(name)
    if val is None or (isinstance(val, str) and not val.strip()):
        w = NotebookWidgets(dbutils=dbutils)
        val = w.get(name)
    if val is None or (isinstance(val, str) and not val.strip()):
        env_val = (
            os.environ.get(f'LANDSEG_{name.upper()}') or
            os.environ.get(name.upper())
        )
        if env_val is not None and env_val.strip():
            val = env_val
        elif default is not None and str(default).strip():
            val = default

    if val is None or (isinstance(val, str) and not val.strip()):
        msg = error_message or (
            f"Required parameter widget '{name}' is not set. "
            f"Please configure widget '{name}' before continuing execution."
        )
        raise MissingRequiredInputError(msg)

    if validator is not None:
        try:
            res = validator(val)
            if res is False:
                msg = error_message or (
                    f"Value '{val}' for required widget '{name}' failed validation."
                )
                raise MissingRequiredInputError(msg)
            if res is not None and res is not True:
                return res
        except MissingRequiredInputError:
            raise
        except Exception as err:
            msg = error_message or (
                f"Validation failed for required widget '{name}' ({val}): {err}"
            )
            raise MissingRequiredInputError(msg) from err

    return val
