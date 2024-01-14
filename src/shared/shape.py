from __future__ import annotations

from torch import Size
from typing import List, Iterable, Any, Tuple
from typing_extensions import Self
from copy import copy
from torch import Size
from math import prod

#TODO: consider using tuple instead, like size does
#or using list, then get the already implemented functionality like the iterator...
#or other option is build on top of size itself, though thats lots of abstraction
class Shape:
	def __init__(self, shape: Tuple[int] | List[int] | Size, fixed: bool = True) -> None:
		self.shape: List[int] = list(shape)
		self.fixed: bool = fixed
	#TODO: find better name for upper
	def upper_length(self) -> int:
		return len(self.shape) - 1 if self.fixed else 0
	def reverse_upper_shape(self, reverse_index: int) -> List[int]:
		return self.shape[-reverse_index:]
	def reverse_lower_product(self, reverse_index: int) -> int:
		return prod(self.shape[:-reverse_index])
	def dimensions(self) -> int:
		return len(self.shape) + 1 if not self.fixed else 0
	def upper_compatible(self, other: Self) -> bool:
		reverse_index = min(self.upper_length(), other.upper_length())
		return self.shape[-reverse_index:] == other.shape[-reverse_index:] 
	def to_fixed(self, dimension: int) -> Shape:
		return Shape([dimension] + self.shape, True)
	def to_size(self) -> Size:
		if self.fixed:
			return Size(self.shape)
		else:
			return Size([-1] + self.shape)
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
		return None if small.reverse_upper_shape(reverse_index) != big.reverse_upper_shape(reverse_index) else common
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
		return None if small.reverse_upper_shape(reverse_index) != big.reverse_upper_shape(reverse_index) else common
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
	def __init__(self, lower: Size | List[int] | int = Size(), upper: Size | List[int] | int = Size()) -> None:
		lower = Size([lower]) if isinstance(lower, int) else Size(lower)
		upper = Size([upper]) if isinstance(upper, int) else Size(upper)
		if len(lower) != len(upper):
			raise Exception("bound dimensions do not match")
		for lower_bound, upper_bound in zip(lower, upper):
			if lower_bound > upper_bound:
				raise Exception("lower bound greater than upper")
			if lower_bound <= 0:
				raise Exception("lower bound less than 1")
		self.upper: Size = upper
		self.lower: Size = lower 
	def __contains__(self, shape: Size) -> bool:
		if len(shape) != len(self.lower):
			return False
		for lower_bound, upper_bound, i in zip(self.lower, self.upper, shape):
			if i < lower_bound or i > upper_bound:
				return False
		return True
	def __len__(self) -> int:
		return len(self.lower)
	def __str__(self) -> str:
		return f"Bound({self.lower}, {self.upper})"
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
