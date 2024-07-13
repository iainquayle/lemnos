from __future__ import annotations

from typing import Iterable, Any

from math import prod
from copy import copy

from abc import ABC as Abstract, abstractmethod
from dataclasses import dataclass

import math

#rules:
#	if no remaining open dims
#		dims to the right must be the same, dims to the left must be prod the same
#	if one constrained shape
#		dims to right must be same, choose or fit the shape that is the correct size
#	if remaining open dims
#		dims to the right must be the same

class Shape(Abstract):
	__slots__ = ("_shape", "_product_cache")
	def __init__(self, *shape: int) -> None:
		self._shape: tuple[int, ...] = tuple(shape)
		self._product_cache: int = max(prod(self._shape), 1)
	@abstractmethod
	def upper_length(self) -> int:
		pass
	def reverse_upper_equal(self, reverse_index: int, other: Shape) -> bool:
		for i in range(1, reverse_index + 1):
			if self._shape[-i] != other._shape[-i]:
				return False
		return True
	def upper_equal(self, other: Shape) -> bool:
		reverse_index = min(self.upper_length(), other.upper_length())
		return self.reverse_upper_equal(reverse_index, other)
	def reverse_lower_product(self, reverse_index: int) -> int:
		return prod(self._shape[:-reverse_index])
	@abstractmethod
	def dimensionality(self) -> int:
		pass
	@abstractmethod
	def to_locked(self, dimension: int) -> LockedShape:
		pass
	@abstractmethod
	def to_open(self) -> OpenShape:
		pass
	def is_locked(self) -> bool:
		return isinstance(self, LockedShape)
	@abstractmethod
	def squash(self, dimensionality: int) -> Shape:
		pass
	def compatible(self, other: Shape) -> bool:
		return self.common(other) is not None
	@abstractmethod
	def common(self, other: Shape) -> Shape | None:
		pass
	def common_lossless(self, other: Shape) -> Shape | None:
		return self.common(other) if self.dimensionality() > other.dimensionality() else other.common(self)
	@staticmethod
	def reduce_common_lossless(shapes: Iterable[Shape]) -> Shape | None:
		common = OpenShape()
		for shape in shapes:
			common = common.common_lossless(shape)
			if common is None:
				return None
		return common
	def __getitem__(self, index: int) -> int:
		return self._shape[index]
	def __len__(self) -> int:
		return len(self._shape)
	def __iter__(self) -> Iterable[int]:
		return iter(self._shape)
	@abstractmethod
	def __eq__(self, other: Any) -> bool:
		pass
	@abstractmethod
	def __copy__(self) -> Shape:
		pass
	def get_product(self) -> int:
		return self._product_cache
	def __repr__(self) -> str:
		return str(self)

class LockedShape(Shape):
	def __init__(self, *shape: int) -> None:
		if len(shape) == 0:
			raise Exception("locked shape cannot be empty")
		super().__init__(*shape)
	def upper_length(self) -> int:
		return len(self) - 1
	def dimensionality(self) -> int:
		return len(self)
	def upper_difference(self, other: LockedShape) -> int:
		accumulator = 1
		for i in range(1, min(len(self), len(other))):
			accumulator *= abs(self[i] - other[i]) + 1 
		return accumulator
	def to_locked(self, dimension: int) -> LockedShape:
		return self 
	def to_open(self) -> OpenShape:
		return OpenShape(*self._shape[1:])
	def squash(self, dimensionality: int) -> LockedShape:
		if dimensionality >= self.dimensionality():
			return self 
		elif dimensionality > 1:
			return LockedShape(self.reverse_lower_product(dimensionality - 1), *(self._shape[-(dimensionality - 1):]))
		else:
			return LockedShape(self.get_product())
	def common(self, other: Shape) -> Shape | None:
		reverse_index = min(self.upper_length(), other.upper_length())
		if other.is_locked():
			if self.reverse_lower_product(reverse_index) != other.reverse_lower_product(reverse_index):
				return None
		elif self[0] % other.reverse_lower_product(reverse_index) != 0:
				return None
		return self if self.reverse_upper_equal(reverse_index, other) else None 
	def scale(self, scalars: list[float]) -> LockedShape:
		return LockedShape(*[int(scalar * element) for element, scalar in zip(self._shape, scalars)])
	def __eq__(self, other: Any) -> bool:
		return other is not None and isinstance(other, LockedShape) and self._shape == other._shape
	def __copy__(self) -> LockedShape:
		return LockedShape(*self._shape)
	def __str__(self) -> str:
		return f"LS({self._shape})"
		
class OpenShape(Shape):
	def upper_length(self) -> int:
		return len(self)
	def dimensionality(self) -> int:
		return len(self) + 1
	def to_locked(self, dimension: int) -> LockedShape:
		return LockedShape(dimension, *self._shape)
	def to_open(self) -> OpenShape:
		return self
	def squash(self, dimensionality: int) -> OpenShape:
		if dimensionality > self.dimensionality():
			return self
		else:
			return OpenShape(*self._shape[-(dimensionality - 1):])
	def common(self, other: Shape) -> Shape | None:
		common = self
		reverse_index = min(self.upper_length(), other.upper_length())
		if other.is_locked():
			other_product = other.reverse_lower_product(reverse_index)
			self_product = self.reverse_lower_product(reverse_index)
			if other_product % self_product != 0:
				return None
			common = self.to_locked(other_product // self_product)
		return common if self.reverse_upper_equal(reverse_index, other) else None 
	def __eq__(self, other: Any) -> bool:
		return other is not None and isinstance(other, OpenShape) and self._shape == other._shape
	def __copy__(self) -> OpenShape:
		return OpenShape(*self._shape)
	def __str__(self) -> str:
		return f"OS({self._shape})"
	def get_upper_diff(self, other: Shape) -> int:
		return 0

_LOWER_INDEX = 0
_UPPER_INDEX = 1
class ShapeBound:
	__slots__ = ("_bounds")
	def __init__(self, *bounds: tuple[int | None, int | None] | int | None) -> None:
		if len(bounds) == 0:
			raise Exception("bounds cannot be empty")
		self._bounds: list[tuple[int | None, int | None]] = [bound if isinstance(bound, tuple) else (bound, bound) for bound in bounds] 
		for i in range(len(self._bounds)):
			upper, lower = self._bounds[i]
			if (lower is not None and upper is not None and
					lower > upper):
				self._bounds[i] = upper, lower
			if ((lower is not None and lower <= 0) or 
					(upper is not None and upper <= 0)):
				raise Exception("bound less than 1")
	def lower(self, index: int) -> int | None:
		element = self._bounds[index]
		return element[_LOWER_INDEX] if element is not None else None
	def upper(self, index: int) -> int | None:
		element = self._bounds[index]
		return element[_UPPER_INDEX] if element is not None else None
	def clamp(self, shape: Shape) -> Shape:
		if shape.dimensionality() > len(self._bounds):
			raise Exception("shape dimensionality greater than bounds")
		new_shape = list(iter(shape))
		for i in range(1, len(new_shape) + 1):
			new_shape[-i] = self.clamp_value(new_shape[-i], -i)
		return LockedShape(*new_shape) if shape.is_locked() else OpenShape(*new_shape)
	def clamp_value(self, value: int, index: int) -> int:
		lower, upper = self._bounds[index]
		if lower is not None:
			value = max(lower, value)
		if upper is not None:
			value = min(upper, value)
		return value
	def contains_value(self, value: int, index: int) -> bool:
		lower, upper = self._bounds[index]
		if lower is not None and value < lower:
			return False
		if upper is not None and value > upper:
			return False
		if value < 1:
			return False
		return True
	def get_bounds(self) -> list[tuple[int | None, int | None]]:
		return copy(self._bounds)
	def scale(self, scalars: list[float]) -> ShapeBound:
		return ShapeBound(*[(int(scalar * lower) if lower is not None else None, int(scalar * upper) if upper is not None else None) for (lower, upper), scalar in zip(self._bounds, scalars)])
	def __contains__(self, shape: Shape) -> bool:
		for i in range(1, min(len(shape), len(self._bounds)) + 1):
			if not self.contains_value(shape[-i], -i):
				return False
		return True
	def __iter__(self) -> Iterable[tuple[int | None, int | None]]:
		return iter(self._bounds)
	def __getitem__(self, index: int) -> tuple[int | None, int | None] | None:
		return self._bounds[index]
	def __len__(self) -> int:
		return len(self._bounds)
	def __str__(self) -> str:
		return f"ShapeBound({self._bounds})"
	def __repr__(self) -> str:
		return str(self)


@dataclass(frozen=True)
class ShapeConformance:
	shape: Shape
	divisor: int
	def common(self, other: ShapeConformance) -> ShapeConformance | None:
		if (shape := self.shape.common_lossless(other.shape)) is not None:
			return ShapeConformance(shape, math.lcm(self.divisor, other.divisor))
		else:
			return None
	def common_divisor(self, divisor: int) -> ShapeConformance:
		return ShapeConformance(self.shape, math.lcm(self.divisor, divisor))
	def common_shape(self, shape: Shape) -> ShapeConformance | None:
		return self.common(ShapeConformance(shape, 1))
	def get_divisor(self, other_divisor: int) -> int:
		return math.lcm(self.divisor, other_divisor)
	def is_compatible(self, shape: LockedShape) -> bool:
		return self.shape.compatible(shape) and shape[0] % self.divisor == 0
	
