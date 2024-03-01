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
	def validate_dimensionality(self, dimensionality: int) -> bool:
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
		def fill_conv_tuple(val: Tuple | int) -> Tuple:
			return val if isinstance(val, tuple) else tuple([val])
		self._kernel: Tuple = fill_conv_tuple(kernel)
		self._stride: Tuple = fill_conv_tuple(stride)
		self._dilation: Tuple = fill_conv_tuple(dilation)
		self._padding: Tuple = fill_conv_tuple(padding)
		if ((len(self._kernel) != len(self._stride) and min(len(self._kernel), len(self._stride)) != 1)
		  		or (len(self._stride) != len(self._dilation) and min(len(self._stride), len(self._dilation)) != 1)
				or (len(self._dilation) != len(self._padding) and min(len(self._dilation), len(self._padding)) != 1)):
			raise Exception("kernel, stride, dilation, padding must all have the same length and be one less than shape_bounds")
		self.depthwise: bool = depthwise
	def output_dim_to_input_dim(self, output_shape: LockedShape, i: int) -> int:
		i -= 1
		return (output_shape[i + 1] - 1) * self._stride[i] + (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1)) - self._padding[i] * 2
	def input_dim_to_output_dim(self, input_shape: LockedShape, i: int) -> int:
		i -= 1
		return ((input_shape[i + 1] + self._padding[i] * 2) - (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1))) // self._stride[i] + 1
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, shape_bounds: ShapeBound, index: Index) -> LockedShape | None:
		#consider making it return none is the output shape does not result in a perfect fit for the input shape
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
	def validate_dimensionality(self, dimensionality: int) -> bool:
		def resize_conv_tuple(val: Tuple, length: int) -> Tuple:
			return val + (val[-1],) * (length - len(val))
		self._kernel = resize_conv_tuple(self._kernel, dimensionality - 1)
		self._stride = resize_conv_tuple(self._stride, dimensionality - 1)
		self._dilation = resize_conv_tuple(self._dilation, dimensionality - 1)
		self._padding = resize_conv_tuple(self._padding, dimensionality - 1)
		return dimensionality >= 2
	def get_init_src(self, shape_in: LockedShape, shape_out: LockedShape) -> str:
		return conv_(shape_in, shape_out, self._kernel, self._stride, self._padding, shape_in[0] if self.depthwise else 1)
