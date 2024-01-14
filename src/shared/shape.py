from __future__ import annotations

from torch import Size
from typing import List, Iterable, Any
from typing_extensions import Self
from copy import copy
from torch import Size
from math import prod

class Shape:
	def __init__(self, shape: List[int] | Size, fixed: bool = True) -> None:
		self.shape: List[int] = list(shape)
		self.fixed: bool = fixed
	#TODO: find better name for upper
	def upper_length(self) -> int:
		return len(self.shape) - 1 if self.fixed else 0
	def reverse_upper_shape(self, index: int) -> List[int]:
		return self.shape[-index:]
	def reverse_lower_product(self, index: int) -> int:
		return prod(self.shape[:-index])
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
	def common_large(self, other: Self) -> Shape | None:
		self_larger = len(self) > len(other) or (len(self) == len(other) and self.dimensions() > other.dimensions())
		big = self if self_larger else other 
		small = other if self_larger else self 
		common = copy(big)
		reverse_index = min(self.upper_length(), other.upper_length())
		if big.fixed:
			if small.fixed and prod(small.shape[:-reverse_index]) != prod(big.shape[:-reverse_index]):
					return None
		else:
			if small.fixed:
				big_product = prod(big.shape[:-reverse_index])
				if big_product % small.shape[0] != 0:
					return None
				common = big.to_fixed(big_product // small.shape[0])
		return None if small.shape[-reverse_index:] != big.shape[-reverse_index:] else common
	def common_small(self, other: Self) -> Shape | None:
		self_smaller = len(self) < len(other) or (len(self) == len(other) and self.dimensions() < other.dimensions())
		small = self if self_smaller else other 
		big = other if self_smaller else self 
		common = copy(small)
		reverse_index = min(self.upper_length(), other.upper_length())
		if small.fixed:
			if big.fixed and prod(small.shape[:-reverse_index]) != prod(big.shape[:-reverse_index]):
					return None
		else:
			if big.fixed:
				common = small.to_fixed(prod(big.shape[:-reverse_index]))
		return None if small.shape[-reverse_index:] != big.shape[-reverse_index:] else common
	@staticmethod
	def common_collection_large(shapes: Iterable[Shape]) -> Shape | None:
		shapes_iter = iter(shapes)
		common = next(shapes_iter)
		if common is None:
			raise Exception("cannot reduce empty collection")
		for shape in shapes_iter:
			common = common.common_large(shape)
			if common is None:
				return None
		return common
	@staticmethod
	def common_collection_small(shapes: Iterable[Shape]) -> Shape | None:
		shapes_iter = iter(shapes)
		common = next(shapes_iter)
		if common is None:
			raise Exception("cannot reduce empty collection")
		for shape in shapes_iter:
			common = common.common_small(shape)
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
