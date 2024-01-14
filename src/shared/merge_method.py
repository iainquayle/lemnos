from __future__ import annotations

from typing import List
from abc import ABC as Abstract, abstractmethod
from math import prod
from src.shared.shape import Shape 

class MergeMethod(Abstract):
	#currently can only take shapes of a higher dimension or same
	#this could be changed by checking that the bottom dimension can be split into the lower dimensions of higher dimension sibling shapes
	#come to think of it, it may only be possible if restricted to a jump of 1 dimension 
	#	or vice versa, and new higher dimensions shapes lower dimensions produce the same size
	#		this would be harder, but the other way would be viable
	@abstractmethod
	def get_conformance_shape(self, sibling_shapes: List[Shape], shape_bounds: Bound) -> Shape:
		pass
	@abstractmethod
	def get_total_merged_size(self, shapes: List[Shape]) -> int:
		pass
	@abstractmethod
	def get_merge_src(self, registers: List[str]) -> str | None:
		pass
class Concat(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Shape], shape_bounds: Bound) -> Shape:
		if len(sibling_shapes) == 0:
			return Shape([], False)
		else:
			shape_list = list(sibling_shapes[0])
			copy_cutoff = len(shape_list) - len(shape_bounds) + 1
			shape_list = shape_list[copy_cutoff:]
			return Shape(len(shape_bounds), Shape(shape_list))
	def get_total_merged_size(self, shapes: List[Shape]) -> int:
		return sum([prod(shape) for shape in shapes])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return f"torch.cat([{', '.join(registers)}], dim=1)"
class Add(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[Shape], shape_bounds: Bound) -> Shape:
		if len(sibling_shapes) == 0:
			return Shape(len(shape_bounds), Shape())
		else:
			shape_list = list(sibling_shapes[0])
			copy_cutoff = len(shape_list) - len(shape_bounds) + 1
			shape_list = [prod(shape_list[:copy_cutoff])] + shape_list[copy_cutoff:]
			return Shape(len(shape_bounds), Shape(shape_list))
	def get_total_merged_size(self, shapes: List[Shape]) -> int:
		return prod(shapes[0])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return f"{' + '.join(registers)}"
