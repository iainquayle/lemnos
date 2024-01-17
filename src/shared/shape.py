from __future__ import annotations

from torch import Size
from typing import List, Iterable, Tuple
from typing_extensions import Self
from copy import copy
from torch import Size
from math import prod
from abc import ABC as Abstract, abstractmethod
#not extending:
#	size, as it is just tuple
#	tuple, because init doesnt play nice when adding other parameters, and makes manipulation harder
#TODO: probably want to optim some of this
#TODO: make a proper to src
#TODO: consider making more for loop implementing functions rather than slice, as it may perform better 
#rules:
#	if no remaining open dims
#		dims to the right must be the same, dims to the left must be prod the same
#	if one constrained shape
#		dims to right must be same, choose or fit the shape that is the correct size
#	if remaining open dims
#		dims to the right must be the same

#TODO: consider renaming locked to fixed, or sized
#TODO: make prod cache?
#TODO: make validating inits
class Shape(Abstract):
	__slots__ = ("_shape")
	def __init__(self, shape: Tuple[int, ...] | List[int] | Size) -> None:
		self._shape: List[int] = list(shape) if not isinstance(shape, List) else shape
	@staticmethod
	@abstractmethod
	def new(*values: int) -> Shape:
		pass
	@abstractmethod
	def upper_length(self) -> int:
		pass
	def reverse_upper_equal(self, reverse_index: int, other: Shape) -> bool:
		for i in range(1, reverse_index + 1):
			if self._shape[-i] != other._shape[-i]:
				return False
		return True
	def reverse_lower_product(self, reverse_index: int) -> int:
		return prod(self._shape[:-reverse_index])
	@abstractmethod
	def dimensionality(self) -> int:
		pass
	def upper_compatible(self, other: Shape) -> bool:
		reverse_index = min(self.upper_length(), other.upper_length())
		return self.reverse_upper_equal(reverse_index, other)
	@abstractmethod
	def to_locked(self, dimension: int) -> LockedShape:
		pass
	@abstractmethod
	def to_open(self) -> OpenShape:
		pass
	@abstractmethod
	def squash(self, dimensionality: int) -> Shape:
		pass
	def compatible(self, other: Shape) -> bool:
		return self.common(other) is not None
	def is_locked(self) -> bool:
		return isinstance(self, LockedShape)
	def common_lossless(self, other: Shape) -> Shape | None:
		return self.common(other) if len(self) > len(other) or (len(self) == len(other) and self.dimensionality() > other.dimensionality()) else other.common(self)
	@abstractmethod
	def common(self, other: Shape) -> Shape | None:
		pass
	@staticmethod
	def reduce_common_lossless(shapes: Iterable[Shape]) -> Shape | None:
		shapes_iter = iter(shapes)
		common = next(shapes_iter)
		if common is None:
			raise Exception("cannot reduce empty collection")
		for shape in shapes_iter:
			common = common.common_lossless(shape)
			if common is None:
				return None
		return common
	@abstractmethod
	def to_tuple(self) -> Tuple[int, ...]:
		pass
	def __getitem__(self, index: int) -> int:
		return self._shape[index]
	def __len__(self) -> int:
		return len(self._shape)
	def __iter__(self) -> Iterable[int]:
		return iter(self._shape)
	@abstractmethod
	def __eq__(self, other: Self) -> bool:
		pass
	@abstractmethod
	def __copy__(self) -> Shape:
		pass

class LockedShape(Shape):
	@staticmethod
	def new(*values: int) -> LockedShape:
		return LockedShape(values)
	def upper_length(self) -> int:
		return len(self) - 1
	def dimensionality(self) -> int:
		return len(self)
	def to_locked(self, dimension: int) -> LockedShape:
		return LockedShape(self._shape)
	def to_open(self) -> OpenShape:
		return OpenShape(self._shape[1:])
	def squash(self, dimensionality: int) -> LockedShape:
		if dimensionality > self.dimensionality():
			return copy(self) 
		else:
			return LockedShape([prod(self._shape[:-(dimensionality - 1)])] + self._shape[-(dimensionality - 1):])
	def common(self, other: Shape) -> Shape | None:
		common = copy(self)
		reverse_index = min(self.upper_length(), other.upper_length())
		if other.is_locked():
			if self.reverse_lower_product(reverse_index) != other.reverse_lower_product(reverse_index):
				return None
		elif self[0] % other.reverse_lower_product(reverse_index) != 0:
				return None
		return common if self.reverse_upper_equal(reverse_index, other) else None 
	def to_tuple(self) -> Tuple[int, ...]:
		return tuple(self._shape)
	def __eq__(self, other: Self) -> bool:
		return other is not None and isinstance(other, LockedShape) and self._shape == other._shape
	def __copy__(self) -> LockedShape:
		return LockedShape(self._shape)

class OpenShape(Shape):
	@staticmethod
	def new(*values: int) -> OpenShape:
		return OpenShape(values)
	def upper_length(self) -> int:
		return len(self)
	def dimensionality(self) -> int:
		return len(self) + 1
	def to_locked(self, dimension: int) -> LockedShape:
		return LockedShape([dimension] + self._shape)
	def to_open(self) -> OpenShape:
		return OpenShape(self._shape)
	def squash(self, dimensionality: int) -> OpenShape:
		if dimensionality > self.dimensionality():
			return copy(self)
		else:
			return OpenShape(self._shape[-(dimensionality - 1):])
	def common(self, other: Shape) -> Shape | None:
		common = copy(self)
		reverse_index = min(self.upper_length(), other.upper_length())
		if other.is_locked():
			other_product = other.reverse_lower_product(reverse_index)
			self_product = self.reverse_lower_product(reverse_index)
			if other_product % self_product != 0:
				return None
			common = self.to_locked(other_product // self_product)
		return common if self.reverse_upper_equal(reverse_index, other) else None 
	def to_tuple(self) -> Tuple[int, ...]:
		return tuple([-1] + self._shape)
	def __eq__(self, other: Self) -> bool:
		return other is not None and isinstance(other, OpenShape) and self._shape == other._shape
	def __copy__(self) -> OpenShape:
		return OpenShape(self._shape)

class Bound:
	def __init__(self, bounds: List[Tuple[int, int]] = []) -> None:
		for i in range(len(bounds)):
			if bounds[i][0] > bounds[i][1]:
				bounds[i] = (bounds[i][1], bounds[i][0])
			if bounds[i][0] <= 0:
				raise Exception("lower bound less than 1")
		self._bounds: List[Tuple[int, int]] = bounds
	def __contains__(self, shape: Shape) -> bool:
		if shape.dimensionality() != len(self._bounds):
			return False
		#TODO: probably nicer way to do this, dont really like using the -1
		for (lower, upper), i in zip(self._bounds, shape.to_tuple()):
			if i != -1 and (i < lower or i > upper):
				return False
		return True
	def __len__(self) -> int:
		return len(self._bounds)
	def __str__(self) -> str:
		return f"Bound({self._bounds})"
	def __repr__(self) -> str:
		return str(self)

class Range:
	def __init__(self, lower: float = 1, upper: float = 1) -> None:
		if upper < lower:
			exit("upper smaller than lower bound")
		self._upper: float = upper
		self._lower: float = lower
	def difference(self) -> int | float:
		return self._upper - self._lower
