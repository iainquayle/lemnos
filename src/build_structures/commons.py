from __future__ import annotations

import torch
from torch import Size
from typing import List

from dataclasses import dataclass

from math import prod

from abc import ABC as Abstract, abstractmethod

@dataclass
class ConformanceShape:
	dimensions: int
	partial_shape: Size
	@staticmethod
	def reduce_collection(conformance_shapes: List[ConformanceShape]) -> ConformanceShape | None:
		if len(conformance_shapes) == 0:
			raise Exception("cannot reduce empty collection")
		else:
			dimensions = 0
			partial_shape = Size()
			for conformance_shape in conformance_shapes:
				i = 0
				while i < len(conformance_shape.partial_shape) and i < len(partial_shape):
					if conformance_shape.partial_shape[-i] != partial_shape[-i]:
						return None
					i += 1
				if len(conformance_shape.partial_shape) > len(partial_shape):
					partial_shape = conformance_shape.partial_shape
				if conformance_shape.dimensions > dimensions:
					dimensions = conformance_shape.dimensions
			return ConformanceShape(dimensions, partial_shape)

class MergeMethod(Abstract):
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
		return ConformanceShape(len(shape_bounds), Size(sibling_shapes[0][1:] if len(sibling_shapes) > 0 else []))
	def get_total_merged_size(self, shapes: List[Size]) -> int:
		return sum([prod(shape) for shape in shapes])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return f"torch.cat([{', '.join(registers)}], dim=1)"
class Add(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Size], shape_bounds: Bound) -> ConformanceShape:
		return ConformanceShape(len(shape_bounds), Size(sibling_shapes[0] if len(sibling_shapes) > 0 else []))
	def get_total_merged_size(self, shapes: List[Size]) -> int:
		return prod(shapes[0])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return f"{' + '.join(registers)}"

class Index:
	MAX_INDEX = 2**16 -1
	def __init__(self, index: int =0) -> None:
		self.set_index(index)
	def set_index(self, index: int) -> None:
		self.index = index % Index.MAX_INDEX
	def get_index(self, max_index: int) -> int:
		return self.index % max_index if max_index > 0 else 0
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
	def from_index(self, index: Index, size: int) -> int:
		return int((self.lower * size) + index.get_index((int)(self.difference() * size)))
