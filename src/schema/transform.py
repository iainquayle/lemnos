from __future__ import annotations

from src.shared import Shape, LockedShape, OpenShape, ShapeBound, Range, Index
from .src_generation import * 

from abc import ABC as Abstract, abstractmethod 
from typing import Tuple

#maybe just dont even use a range class

class Transform(Abstract):
	__slots__ = ["_size_coefficients"]
	#size_delta_coeffs: int | Tuple[int, int] 
	def __init__(self, size_coefficients: Range) -> None:
		self._size_coefficients: Range = size_coefficients
	@abstractmethod
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		pass
	@abstractmethod
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, index: Index) -> LockedShape | None:
		pass
	@abstractmethod
	def get_init_src(self, shape_in: LockedShape, shape_out: LockedShape) -> str:	
		pass


class Conv(Transform):
	__slots__ = ["_size_coefficients", "_merge_method", "_kernel", "_stride", "_dilation", "_padding", "depthwise"]
	def __init__(self,
			size_coefficients: Range,
			kernel: Tuple | int = 1, 
			stride: Tuple | int = 1, 
			dilation: Tuple | int = 1,
			padding: Tuple | int = 0,
			depthwise: bool = False, #TODO: change this to a factor? or somthing else so filter groups can be a different size and a different number of groups
			) -> None:
		super().__init__(size_coefficients)
		self._kernel: _ClampedTuple = _ClampedTuple(kernel)
		self._stride: _ClampedTuple = _ClampedTuple(stride)
		self._dilation: _ClampedTuple = _ClampedTuple(dilation)
		self._padding: _ClampedTuple = _ClampedTuple(padding)
		self.depthwise: bool = depthwise
	def output_dim_to_input_dim(self, output_shape: LockedShape, i: int) -> int:
		i -= 1
		return (output_shape[i + 1] - 1) * self._stride[i] + (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1)) - self._padding[i] * 2
	def input_dim_to_output_dim(self, input_shape: LockedShape, i: int) -> int:
		i -= 1
		return ((input_shape[i + 1] + self._padding[i] * 2) - (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1))) // self._stride[i] + 1
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, index: Index) -> LockedShape | None:
		if len(input_shape) < 2:
			raise ValueError("input shape must have at least 2 dimensions")
		open_shape = OpenShape(*[self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))])
		if output_conformance.is_locked():
			return open_shape.to_locked(output_conformance.get_product() // open_shape.get_product())
		else:
			lower = int(input_shape.get_product() * self._size_coefficients.lower()) // output_conformance.get_product()
			upper = int(input_shape.get_product() * self._size_coefficients.upper()) // output_conformance.get_product()
			return open_shape.to_locked(shape_bounds.clamp_value(index.get_shuffled((lower, upper), 0) , 0))
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (not self.depthwise or shape_out[0] == shape_in[0])
	def get_init_src(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		dimensionality = len(shape_in) - 1
		return conv_(shape_in, shape_out, self._kernel.expand(dimensionality), self._stride.expand(dimensionality), self._padding.expand(dimensionality), shape_in[0] if self.depthwise else 1)

class _ClampedTuple:
	slot = ["_val"]
	def __init__(self, val: Tuple[int, ...] | int) -> None:
		self._val: Tuple[int, ...] = val if isinstance(val, tuple) else (val,)
		if len(self._val) == 0:
			raise ValueError("empty tuple")
	def __getitem__(self, index: int) -> int:
		if index > len(self._val):
			return self._val[-1]
		else:
			return self._val[index]
	def expand(self, dimensionality: int) -> Tuple[int, ...]:
		return self._val + (self._val[-1],) * (dimensionality - len(self._val))
