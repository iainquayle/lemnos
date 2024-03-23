from __future__ import annotations

from ...shared import LockedShape, OpenShape, Shape

from typing import Iterable
from abc import ABC as Abstract, abstractmethod
from functools import reduce

class MergeMethod(Abstract):
	@abstractmethod
	def get_conformance_shape(self, input_shapes: list[LockedShape]) -> Shape:
		pass
	def get_merged_shape(self, input_shapes: Iterable[LockedShape]) -> LockedShape:
		if next(iter(input_shapes)) is None:
			raise Exception("cannot get merged shape from empty input shapes")
		else:
			return self._get_merged_shape(input_shapes)
	@abstractmethod
	def _get_merged_shape(self, input_shapes: Iterable[LockedShape]) -> LockedShape:
		pass
	@abstractmethod
	def get_total_merged_size(self, input_shapes: list[LockedShape]) -> int:
		pass


class Concat(MergeMethod):
	def get_conformance_shape(self, input_shapes: list[LockedShape]) -> Shape:
		if len(input_shapes) == 0:
			return OpenShape()
		else:
			return input_shapes[0].to_open()
	def get_total_merged_size(self, input_shapes: list[LockedShape]) -> int:
		return sum([shape.get_product() for shape in input_shapes])
	def _get_merged_shape(self, input_shapes: Iterable[LockedShape]) -> LockedShape:
		inputs_iter = iter(input_shapes)
		largest_shape = next(inputs_iter) 
		total_size = largest_shape.get_product()
		for shape in inputs_iter:
			if len(shape) > len(largest_shape):
				largest_shape = shape
			total_size += largest_shape.get_product()
		largest_shape = largest_shape.to_open()
		return largest_shape.to_locked(total_size // largest_shape.get_product())

class Sum(MergeMethod):
	def get_conformance_shape(self, input_shapes: list[LockedShape]) -> Shape:
		if len(input_shapes) == 0:
			return OpenShape()
		else:
			return input_shapes[0]
	def get_total_merged_size(self, input_shapes: list[LockedShape]) -> int:
		return input_shapes[0].get_product()
	def _get_merged_shape(self, input_shapes: Iterable[LockedShape]) -> LockedShape:
		return reduce(lambda x, y: x if len(x) > len(y) else y, input_shapes)
