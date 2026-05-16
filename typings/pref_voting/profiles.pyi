from collections.abc import Sequence
from typing import Any

Ranking = tuple[int, ...]

class Profile:
    rankings: list[Ranking]
    num_cands: int
    candidates: list[int]
    num_voters: int

    def __init__(
        self,
        rankings: Sequence[Sequence[int]],
        rcounts: Any | None = None,
        cmap: Any | None = None,
    ) -> None: ...
