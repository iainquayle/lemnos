from __future__ import annotations

from typing import List
from abc import ABC as Abstract, abstractmethod
from math import prod
from src.shared.shape import Shape, LockedShape, OpenShape

class MergeMethod(Abstract):
	@abstractmethod
	def get_conformance_shape(self, sibling_shapes: List[LockedShape]) -> Shape:
		pass
	@abstractmethod
	def get_total_merged_size(self, sibling_shapes: List[LockedShape]) -> int:
		pass
	@abstractmethod
	def get_merge_src(self, registers: List[str]) -> str | None:
		pass
class Concat(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[LockedShape]) -> Shape:
		if len(sibling_shapes) == 0:
			return OpenShape.new()
		else:
			return sibling_shapes[0].to_open()
	def get_total_merged_size(self, shapes: List[LockedShape]) -> int:
		return sum([prod(iter(shape)) for shape in shapes])
	def get_merge_src(self, registers: List[str]) -> str | None:
		return ""
class Add(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[LockedShape]) -> Shape:
		if len(sibling_shapes) == 0:
			return OpenShape.new()
		else:
			return sibling_shapes[0]
	def get_total_merged_size(self, shapes: List[LockedShape]) -> int:
		return prod(iter(shapes[0]))
	def get_merge_src(self, registers: List[str]) -> str | None:
		return ""
