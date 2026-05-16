from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

class KeyedVectors:
    vectors: NDArray[np.floating[Any]]
    key_to_index: dict[str, int]

class Word2Vec:
    vector_size: int
    wv: KeyedVectors

    def __init__(
        self,
        sentences: Iterable[Sequence[str]] | None = None,
        corpus_file: str | None = None,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 3,
        sg: int = 0,
        **kwargs: Any,
    ) -> None: ...

    def save(self, fname: str, **kwargs: Any) -> None: ...

    @classmethod
    def load(cls, fname: str, **kwargs: Any) -> Word2Vec: ...
