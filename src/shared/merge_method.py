from __future__ import annotations

from typing import List, Iterable
from abc import ABC as Abstract, abstractmethod
from math import prod
from src.shared.shape import Shape, LockedShape, OpenShape
from functools import reduce

class MergeMethod(Abstract):
	@abstractmethod
	def get_conformance_shape(self, sibling_shapes: List[LockedShape], dimensionality: int) -> Shape:
		pass
	def get_output_shape(self, sibling_shapes: Iterable[LockedShape], dimensionality: int) -> LockedShape:
		if next(iter(sibling_shapes)) is None:
			raise Exception("cannot get output shape from empty sibling shapes")
		else:
			return self._get_output_shape(sibling_shapes, dimensionality)
	@abstractmethod
	def _get_output_shape(self, sibling_shapes: Iterable[LockedShape], dimensionality: int) -> LockedShape:
		pass
	@abstractmethod
	def get_total_merged_size(self, sibling_shapes: List[LockedShape]) -> int:
		pass
	@abstractmethod
	def get_merge_src(self, registers: List[str]) -> str | None:
		pass
class Concat(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[LockedShape], dimensionality: int) -> Shape:
		if len(sibling_shapes) == 0:
			return OpenShape.new()
		else:
			return sibling_shapes[0].to_open().squash(dimensionality)
	def get_total_merged_size(self, shapes: List[LockedShape]) -> int:
		return sum([prod(iter(shape)) for shape in shapes])
	def _get_output_shape(self, sibling_shapes: Iterable[LockedShape], dimensionality: int) -> LockedShape:
		#TODO: expand so its quicker
		return reduce(lambda x, y: x if len(x) > len(y) else y, sibling_shapes).squash(dimensionality)
	def get_merge_src(self, registers: List[str]) -> str | None:
		return ""
class Add(MergeMethod):
	def get_conformance_shape(self, sibling_shapes: List[LockedShape], dimensionality: int) -> Shape:
		if len(sibling_shapes) == 0:
			return OpenShape.new()
		else:
			return sibling_shapes[0].squash(dimensionality)
	def get_total_merged_size(self, shapes: List[LockedShape]) -> int:
		return prod(iter(shapes[0]))
	def _get_output_shape(self, sibling_shapes: Iterable[LockedShape], dimensionality: int) -> LockedShape:
		siblings_iter = iter(sibling_shapes)
		largest_shape = next(siblings_iter) 
		total_size = prod(iter(largest_shape))
		for shape in siblings_iter:
			if len(shape) > len(largest_shape):
				largest_shape = shape
			total_size += prod(iter(shape))
		largest_shape = largest_shape.to_open().squash(dimensionality)
		return largest_shape.to_locked(total_size // prod(iter(largest_shape)))
	def get_merge_src(self, registers: List[str]) -> str | None:
		return ""
