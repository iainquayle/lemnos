from __future__ import annotations

from typing import Any

import random

class CompileIndex:
	__slots__ = ["_index"]
	def __init__(self, index: int = 0) -> None:
		self._index: int = index
	@staticmethod
	def random() -> CompileIndex:
		return CompileIndex(random.randint(0, 2**31 - 1))
	def get_shuffled(self, bounds: tuple[int, int] | int, salt: int = 0) -> int:
		if isinstance(bounds, int):
			bounds = (0, bounds)
		elif bounds[0] > bounds[1]:
			bounds = (bounds[1], bounds[0])
		return random.Random(self._index + salt).randint(*bounds)
	def get_shuffled_float(self, bounds: tuple[float, float] | float, salt: int = 0) -> float:
		return 0
	def get(self) -> int:
		return self._index
	def __eq__(self, other: Any) -> bool:
		return isinstance(other, CompileIndex) and self._index == other._index
	def __str__(self) -> str:
		return str(self._index)
	def __repr__(self) -> str:
		return f"CompileIndex({self._index})"
