from __future__ import annotations

from .schema_graph import CompilationIndex 
from ..shared import LockedShape
from math import log

class LogGrowth:
	def __init__(self, target: int, base: float, variability: float) -> None:
		raise NotImplemented()
	def __call__(self, shape: LockedShape, index: CompilationIndex) -> float:
		raise NotImplemented()
		
class PowerGrowth:
	__slots__ = ["_exponent", "_variability", "_zero"]
	def __init__(self, target: int, exponent: float, variability: float) -> None:
		if target <= 0:
			raise ValueError("target must be greater than zero")
		if exponent <= 0:
			raise ValueError("Exponent must be greater than zero")
		if variability < 0 or variability > 1:
			raise ValueError("Variability must be between 0 and 1")
		self._exponent: float = exponent
		self._variability: float = variability
		self._zero: int = target 
	def __call__(self, shape: LockedShape, index: CompilationIndex) -> float:
		normalized_position = (shape[0] / self._zero)
		next_position = normalized_position ** self._exponent
		next_position = index.get_shuffled((next_position * (1 - self._variability), next_position * (1 + self._variability)))
		return next_position / normalized_position

class LinearGrowth:
	__slots__ = ["_slope", "_variability"]
	def __init__(self, slope: float, variability: float) -> None:
		if slope <= 0:
			raise ValueError("Slope must be greater than zero")
		if variability < 0 or variability > 1:
			raise ValueError("Variability must be between 0 and 1")
		self._slope: float = slope
		self._variability: float = variability
	def __call__(self, shape: LockedShape, index: CompilationIndex) -> float:
		return index.get_shuffled((self._slope * (1 - self._variability), self._slope * (1 + self._variability)))

class InvertedParabolicGrowth:
	__slots__ = ["_target", "_variability"]
	def __init__(self, target: int, variability: float) -> None:
		if target <= 0:
			raise ValueError("Intercept must be greater than zero")
		if variability < 0 or variability > 1:
			raise ValueError("Variability must be between 0 and 1")
		self._target: int = target
		self._variability: float = variability
	def __call__(self, shape: LockedShape, index: CompilationIndex) -> float:
		normalized_position = (shape[0] / self._target)
		next_position = -((normalized_position - 1) ** 2) + 1
		next_position = index.get_shuffled((next_position * (1 - self._variability), next_position * (1 + self._variability)))
		return next_position / normalized_position
