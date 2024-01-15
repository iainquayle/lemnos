from __future__ import annotations

from torch import Size
from typing import List, Iterable, Any, Tuple
from typing_extensions import Self
from copy import copy
from torch import Size
from math import prod

#not extending:
#	size, as it is just tuple
#	tuple, because init doesnt play nice when adding other parameters, and makes manipulation harder
#TODO: probably want to optim some of this
class Shape:
	def __init__(self, shape: Tuple[int, ...] | List[int] | Size, fixed: bool = True) -> None:
		self.shape: List[int] = list(shape) if not isinstance(shape, List) else shape
		self.fixed: bool = fixed
	@staticmethod
	def new_fixed(shape: Tuple[int, ...] | List[int] | Size) -> Shape:
		return Shape(shape, True)
	@staticmethod
	def new_unfixed(shape: Tuple[int, ...] | List[int] | Size) -> Shape:
		return Shape(shape, False)
	def upper_length(self) -> int:
		return len(self.shape) - 1 if self.fixed else 0
	def reverse_upper_equal(self, reverse_index: int, other: Self) -> bool:
		return self.shape[-reverse_index:] == other.shape[-reverse_index:]
	def reverse_lower_product(self, reverse_index: int) -> int:
		return prod(self.shape[:-reverse_index])
	def dimensions(self) -> int:
		return len(self.shape) + 1 if not self.fixed else 0
	def upper_compatible(self, other: Self) -> bool:
		reverse_index = min(self.upper_length(), other.upper_length())
		return self.reverse_upper_equal(reverse_index, other)
	def to_fixed(self, dimension: int) -> Shape:
		return Shape(([] if self.fixed else [dimension]) + self.shape, True)
	def to_unfixed(self) -> Shape:
		return Shape(self.shape[1 if self.fixed else 0:], False)
	def to_tuple(self) -> Tuple[int, ...]:
		return tuple(([] if self.fixed else [-1]) + self.shape)
	#rules:
	#	if no remaining open dims
	#		dims to the right must be the same, dims to the left must be prod the same
	#	if one constrained shape
	#		dims to right must be same, choose or fit the shape that is the correct size
	#	if remaining open dims
	#		dims to the right must be the same
	#TODO: consider making more for loop implementing functions rather than slice, as it may perform better 
	def common_large(self, other: Self) -> Shape | None:
		self_larger = len(self) > len(other) or (len(self) == len(other) and self.dimensions() > other.dimensions())
		big = self if self_larger else other 
		small = other if self_larger else self 
		common = copy(big)
		reverse_index = min(self.upper_length(), other.upper_length())
		if big.fixed:
			if small.fixed and small.reverse_lower_product(reverse_index) != big.reverse_lower_product(reverse_index):
					return None
		else:
			if small.fixed:
				big_product = big.reverse_lower_product(reverse_index)
				if big_product % small.shape[0] != 0:
					return None
				common = big.to_fixed(big_product // small.shape[0])
		return None if big.reverse_upper_equal(reverse_index, small) else common
	def common_small(self, other: Self) -> Shape | None:
		self_smaller = len(self) < len(other) or (len(self) == len(other) and self.dimensions() < other.dimensions())
		small = self if self_smaller else other 
		big = other if self_smaller else self 
		common = copy(small)
		reverse_index = min(self.upper_length(), other.upper_length())
		if small.fixed:
			if big.fixed and small.reverse_lower_product(reverse_index) != big.reverse_lower_product(reverse_index):
					return None
		else:
			if big.fixed:
				common = small.to_fixed(big.reverse_lower_product(reverse_index))
		return None if big.reverse_upper_equal(reverse_index, small) else common
	@staticmethod
	def reduce_common_large(shapes: Iterable[Shape]) -> Shape | None:
		shapes_iter = iter(shapes)
		common = next(shapes_iter)
		if common is None:
			raise Exception("cannot reduce empty collection")
		for shape in shapes_iter:
			common = common.common_large(shape)
			if common is None:
				return None
		return common
	def __copy__(self) -> Shape:
		return Shape(self.shape, self.fixed)
	def __getitem__(self, index: int) -> int:
		return self.shape[index]
	def __len__(self) -> int:
		return len(self.shape)
	def __iter__(self) -> Iterable[int]:
		return iter(self.shape)
	def __eq__(self, other: Any) -> bool:
		return not isinstance(other, Self) and self.shape == other.shape and self.fixed == other.fixed

class Bound:
	def __init__(self, bounds: List[Tuple[int, int]]) -> None:
		for i in range(len(bounds)):
			if bounds[i][0] > bounds[i][1]:
				bounds[i] = (bounds[i][1], bounds[i][0])
			if bounds[i][0] <= 0:
				raise Exception("lower bound less than 1")
		self.bounds: List[Tuple[int, int]] = bounds
	def __contains__(self, shape: Shape) -> bool:
		if shape.dimensions() != len(self.bounds):
			return False
		#TODO: probably nicer way to do this, dont really like using the -1
		for (lower, upper), i in zip(self.bounds, shape.to_tuple()):
			if i != -1 and (i < lower or i > upper):
				return False
		return True
	def __len__(self) -> int:
		return len(self.bounds)
	def __str__(self) -> str:
		return f"Bound({self.bounds})"
	def __repr__(self) -> str:
		return str(self)

class Range:
	def __init__(self, lower: float = 1, upper: float = 1) -> None:
		if upper < lower:
			exit("upper smaller than lower bound")
		self.upper: float = upper
		self.lower: float = lower
	def difference(self) -> int | float:
		return self.upper - self.lower
