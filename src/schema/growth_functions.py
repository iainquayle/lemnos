from __future__ import annotations

from .compile_index import CompileIndex
from ..shared import LockedShape

class PowerGrowth:
	def __init__(self, intercept: int, exponent: float, variability: float) -> None:
		if intercept <= 0:
			raise ValueError("intercept must be greater than zero")
		if exponent <= 0:
			raise ValueError("Exponent must be greater than zero")
		if variability < 0 or variability > 1:
			raise ValueError("Variability must be between 0 and 1")
		self._exponent: float = exponent
		self._variability: float = variability
		self._zero: int = intercept 
	def __call__(self, shape: LockedShape, index: CompileIndex) -> float:
		center = 1 / ((shape[0] / self._zero) ** self._exponent)
		return index.get_shuffled((center * (1 - self._variability), center * (1 + self._variability)))
class LinearGrowth:
	def __init__(self, slope: float, variability: float) -> None:
		if slope <= 0:
			raise ValueError("Slope must be greater than zero")
		if variability < 0 or variability > 1:
			raise ValueError("Variability must be between 0 and 1")
		self._slope: float = slope
		self._variability: float = variability
	def __call__(self, shape: LockedShape, index: CompileIndex) -> float:
		return index.get_shuffled((self._slope * (1 - self._variability), self._slope * (1 + self._variability)))
