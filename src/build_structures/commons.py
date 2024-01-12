from __future__ import annotations

import torch
from torch import Size
from typing import List

from copy import deepcopy
from dataclasses import dataclass

from math import prod

from abc import ABC as Abstract, abstractmethod

#TODO: look for some more mathimatical way to deal with fully vs partially constrained shapes
#	may allow for the removal of a bunch of branching 

class ConformanceShape:
	def __init__(self, dimensions: int, partial_shape: Size) -> None:
		if len(partial_shape) > dimensions:
			raise Exception("partial shape greater than dimensions")
		self.dimensions = dimensions
		self.partial_shape = partial_shape
	def fully_constrained(self) -> bool:
		return len(self.partial_shape) == self.dimensions
	def __eq__(self, other: ConformanceShape) -> bool:
		return self.dimensions == other.dimensions and self.partial_shape == other.partial_shape
	def __str__(self) -> str:
		return f"ConShape({self.dimensions}, {self.partial_shape})"
	def __repr__(self) -> str:
		return str(self)
	def common(self, other: ConformanceShape) -> ConformanceShape | None:
		#rules:
		#	if no remaining open dims
		#		dims to the right must be the same
		#		remaining dims to the left must be product the same
		#	if one constrained shape
		#		take it if larger or fit the larger unconstrained to it, make it constrained
		#	if remaining open dims
		#		dims to the right must be the same, take the larger
		small = self 
		big = other 
		if ((len(small.partial_shape) > len(big.partial_shape))
				or (len(small.partial_shape) == len(big.partial_shape) 
				and small.dimensions > big.dimensions)):
			small = other 
			big = self 
		common = big
		shape_equivilence_cutoff = len(small.partial_shape) 
		if big.fully_constrained():
			if small.fully_constrained():
				shape_equivilence_cutoff -= 1
				if prod(small.partial_shape[:-shape_equivilence_cutoff]) != prod(big.partial_shape[:-shape_equivilence_cutoff]):
					return None
		else:
			if small.fully_constrained():
				shape_equivilence_cutoff -= 1
				big_product = prod(big.partial_shape[:-shape_equivilence_cutoff])
				if big_product % small.partial_shape[0] != 0:
					return None
				common = ConformanceShape(big.dimensions, Size([small.partial_shape[0] // big_product] + list(big.partial_shape))) 
		return None if small.partial_shape[-shape_equivilence_cutoff:] != big.partial_shape[len(big.partial_shape) - shape_equivilence_cutoff:] else deepcopy(common)
	def compatible(self, other: ConformanceShape) -> bool:
		return self.common(other) is not None
	@staticmethod
	def reduce_collection(conformance_shapes: List[ConformanceShape]) -> ConformanceShape | None:
		if len(conformance_shapes) == 0:
			raise Exception("cannot reduce empty collection")
		else:
			shapes_iter = iter(conformance_shapes)
			common = next(shapes_iter)
			for shape in shapes_iter:
				common = common.common(shape)
				if common is None:
					return None
			return common

class MergeMethod(Abstract):
	#currently can only take shapes of a higher dimension or same
	#this could be changed by checking that the bottom dimension can be split into the lower dimensions of higher dimension sibling shapes
	#come to think of it, it may only be possible if restricted to a jump of 1 dimension 
	#	or vice versa, and new higher dimensions shapes lower dimensions produce the same size
	#		this would be harder, but the other way would be viable
	@abstractmethod
	def get_conformance_shape(self, sibling_shapes: List[Size], shape_bounds: Bound) -> ConformanceShape:
		pass
	@abstractmethod
	def get_total_merged_size(self, shapes: List[Size]) -> int:
		pass
	@abstractmethod
	def get_merge_src(self, registers: List[str]) -> str | None:
		pass
class Concat(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Size], shape_bounds: Bound) -> ConformanceShape:
		if len(sibling_shapes) == 0:
			return ConformanceShape(len(shape_bounds), Size())
		else:
			shape_list = list(sibling_shapes[0])
			copy_cutoff = len(shape_list) - len(shape_bounds) + 1
			shape_list = shape_list[copy_cutoff:]
			return ConformanceShape(len(shape_bounds), Size(shape_list))
	def get_total_merged_size(self, shapes: List[Size]) -> int:
		return sum([prod(shape) for shape in shapes])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return f"torch.cat([{', '.join(registers)}], dim=1)"
class Add(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Size], shape_bounds: Bound) -> ConformanceShape:
		if len(sibling_shapes) == 0:
			return ConformanceShape(len(shape_bounds), Size())
		else:
			shape_list = list(sibling_shapes[0])
			copy_cutoff = len(shape_list) - len(shape_bounds) + 1
			shape_list = [prod(shape_list[:copy_cutoff])] + shape_list[copy_cutoff:]
			return ConformanceShape(len(shape_bounds), Size(shape_list))
	def get_total_merged_size(self, shapes: List[Size]) -> int:
		return prod(shapes[0])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return f"{' + '.join(registers)}"

class Index:
	MAX_INDEX = 2**16 -1
	def __init__(self, index: int =0) -> None:
		self.index = index
	def to_int(self, mod_factor: int) -> int:
		return self.index % mod_factor if mod_factor > 0 else 0
	def as_ratio(self) -> float:
		return self.index / Index.MAX_INDEX

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
