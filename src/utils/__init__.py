'''Simple top-level namesapce for utilities.'''

from .funcs import(
    get_timestamp
)
from .logger import Logger

__all__ = [
    'Logger',
    'get_timestamp',
]
