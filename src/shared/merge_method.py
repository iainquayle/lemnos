from __future__ import annotations

from typing import List
from abc import ABC as Abstract, abstractmethod
from math import prod
from src.shared.shape import Shape 

class MergeMethod(Abstract):
	@abstractmethod
	def get_conformance_shape(self, sibling_shapes: List[Shape]) -> Shape:
		pass
	@abstractmethod
	def get_total_merged_size(self, shapes: List[Shape]) -> int:
		pass
	@abstractmethod
	def get_merge_src(self, registers: List[str]) -> str | None:
		pass
class Concat(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Shape]) -> Shape:
		if len(sibling_shapes) == 0:
			return Shape.unfixed()
		else:
			return sibling_shapes[0].to_unfixed()
	def get_total_merged_size(self, shapes: List[Shape]) -> int:
		return sum([prod(iter(shape)) for shape in shapes])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return ""
class Add(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Shape]) -> Shape:
		if len(sibling_shapes) == 0:
			return Shape.unfixed()
		else:
			return sibling_shapes[0]
	def get_total_merged_size(self, shapes: List[Shape]) -> int:
		return prod(iter(shapes[0]))
	def get_merge_src(self, registers: List[str]) -> str | None:
		return ""
