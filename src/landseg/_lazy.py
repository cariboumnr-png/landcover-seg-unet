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
Lazy attribute resolution for `landseg` package namespaces.

Builds the module-level `__getattr__` and `__dir__` hooks (PEP 562)
that package `__init__.py` files use to expose public symbols without
importing their submodules eagerly.

Public APIs:
    - `attach`: Build lazy `__getattr__` and `__dir__` for a package.

Example:
    >>> import typing
    >>> import landseg._lazy as _lazy
    >>> __all__ = [
    ...     'ClassA',
    ...     'func_a',
    ...     'func_b',
    ... ]
    >>> if typing.TYPE_CHECKING:
    ...     from .mod_a import (
    ...         ClassA,
    ...         func_a,
    ...     )
    ...     from .mod_b import func_b
    >>> __getattr__, __dir__ = _lazy.attach(__name__, {
    ...     '.mod_a': [
    ...         'ClassA',
    ...         'func_a',
    ...     ],
    ...     '.mod_b': ['func_b'],
    ... })
'''

# standard imports
import importlib
import sys
import typing


# ----- public functions
def attach(
    package: str,
    exports: typing.Mapping[str, typing.Iterable[str]],
    *,
    submodules: typing.Iterable[str] = (),
) -> tuple[typing.Callable[[str], typing.Any], typing.Callable[[], list[str]]]:
    '''
    Build lazy `__getattr__` and `__dir__` hooks for a package.

    Args:
        package:
            Dotted name of the calling package, i.e. `__name__`.
        exports:
            Relative source module (e.g. `'.paths'`) mapped to the
            symbols it provides.
        submodules:
            Relative submodule names exposed as attributes themselves.

    Returns:
        tuple[typing.Callable[[str], typing.Any], typing.Callable[[], list[str]]]:
            The `__getattr__` and `__dir__` hooks for the package.

    Raises:
        ValueError:
            If a symbol is mapped to more than one source module.
    '''
    def __getattr__(name: str) -> typing.Any:
        '''Resolve a lazy symbol or submodule on access.'''
        if name in owner:
            module = importlib.import_module(owner[name], package)
            return getattr(module, name)
        if name in lazy_mods:
            return importlib.import_module(f'.{name}', package)
        raise AttributeError(f'module {package!r} has no attribute {name!r}')

    def __dir__() -> list[str]:
        '''List eager globals plus lazy symbols and submodules.'''
        return sorted({*namespace, *owner, *lazy_mods})

    owner: dict[str, str] = {}
    for module, names in exports.items():
        for name in names:
            if name in owner:
                raise ValueError(f'{package}: {name!r} exported twice')
            owner[name] = module
    lazy_mods = frozenset(submodules)
    namespace = sys.modules[package].__dict__

    return __getattr__, __dir__
