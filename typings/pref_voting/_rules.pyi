from collections.abc import Callable
from typing import Any

from pref_voting.profiles import Profile

Rule = Callable[[Profile], Any]
