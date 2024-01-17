from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Module, Identity

from src.pattern.commons import Index
from src.shared.shape import Shape, LockedShape, OpenShape, Bound, Range
from src.shared.merge_method import MergeMethod, Concat, Add
from abc import ABC as Abstract, abstractmethod 
from typing import List, Tuple

from math import prod

#TODO: make tokens for constants, ie conv, bn, rbrack, comma
#or make functions, ie then can just use param list
#to expand:
#	get mould shape (dont need to validate, as validation happens when a new node is created and its shape made)
#	fetch conformance shapes from children
#		fetch from all possible children in a transition group
#		conforming shape hold the total size of a conforming shape, and the upper shape that is required
#		requires the dims and the siblings
#	generate conforming shape, or none, from these
class BaseParameters(Abstract):
	def __init__(self) -> None:
		self._shape_bounds = Bound([]) 
		self._merge_method = Concat() 
	def validate_output_shape(self, shape_in: LockedShape, shape_out: Shape) -> bool:
		return self.validate_output_shape_transform(shape_in, shape_out) and shape_out in self._shape_bounds
	@abstractmethod
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: Shape) -> bool:
		pass
	def get_conformance_shape(self, sibling_shapes: List[LockedShape]) -> Shape:
		return self._merge_method.get_conformance_shape(sibling_shapes, self.dimensionality())
	def get_mould_and_output_shapes(self, parent_shapes: List[LockedShape], output_conformance: Shape, index: Index = Index()) -> Tuple[LockedShape, LockedShape] | None:
		mould_shape = self._merge_method.get_output_shape(parent_shapes, self.dimensionality())
		output_shape = self.get_output_shape(mould_shape, output_conformance, index)
		return None if output_shape is None or output_shape not in self._shape_bounds else (mould_shape, output_shape)
	@abstractmethod
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, index: Index = Index()) -> LockedShape | None:
		pass
	def dimensionality(self) -> int:
		return len(self._shape_bounds)
	
class IdentityParameters(BaseParameters):
	def __init__(self, shape_bounds: Bound, merge_method: MergeMethod) -> None:
		self._shape_bounds = shape_bounds
		self._merge_method = merge_method
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		return shape_in == shape_out
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, index: Index = Index()) -> LockedShape | None:
		return input_shape if output_conformance.compatible(input_shape) else None

#TODO: move to commons
def auto_fill_tuple(val: Tuple | int, bounds: Bound) -> Tuple:
	return val if isinstance(val, tuple) else tuple([val] * (len(bounds) - 1))
class ConvParameters(BaseParameters):
	def __init__(self,
			shape_bounds: Bound, 
			size_coefficents: Range,
			merge_method: MergeMethod,
			kernel: Tuple | int = 1, 
			stride: Tuple | int = 1, 
			dilation: Tuple | int = 1,
			padding: Tuple | int = 0,
			depthwise: bool = False,
			) -> None:
		if len(shape_bounds) < 2:
			raise Exception("shape_bounds must have at least two dimensions")
		self._shape_bounds = shape_bounds
		self._size_coefficents = size_coefficents 
		self._merge_method = merge_method
		self._kernel: Tuple = auto_fill_tuple(kernel, shape_bounds)
		self._stride: Tuple = auto_fill_tuple(stride, shape_bounds)
		self._dilation: Tuple = auto_fill_tuple(dilation, shape_bounds)
		self._padding: Tuple = auto_fill_tuple(padding, shape_bounds)
		if (len(self._kernel) != len(self._stride) 
		  		or len(self._stride) != len(self._dilation) 
		  		or len(self._dilation) != len(self._padding) 
		  		or len(self._padding) != len(self._shape_bounds) - 1):
			raise Exception("kernel, stride, dilation, padding must all have the same length and be one less than shape_bounds")
		self.depthwise: bool = depthwise
	def output_dim_to_input_dim(self, output_shape: LockedShape, i: int) -> int:
		i -= 1
		return (output_shape[i + 1] - 1) * self._stride[i] + (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1)) - self._padding[i] * 2
	def input_dim_to_output_dim(self, input_shape: LockedShape, i: int) -> int:
		i -= 1
		return ((input_shape[i + 1] + self._padding[i] * 2) - (self._kernel[i] * self._dilation[i] - (self._dilation[i] - 1))) // self._stride[i] + 1
	def get_output_shape(self, input_shape: LockedShape, output_conformance: Shape, index: Index = Index()) -> LockedShape | None:
		open_shape = OpenShape([self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))])
		if output_conformance.compatible(open_shape): 
			if output_conformance.is_locked():
				return open_shape.to_locked(output_conformance.get_product() // open_shape.get_product())
			else:
				lower = max(self._shape_bounds.lower(0), int(input_shape.get_product() * self._size_coefficents.lower()) // output_conformance.get_product())
				upper = min(self._shape_bounds.upper(0), int(input_shape.get_product() * self._size_coefficents.upper()) // output_conformance.get_product())
				return open_shape.to_locked(index.to_int(upper - lower) + lower)
		else:
			return None
	def validate_output_shape_transform(self, shape_in: LockedShape, shape_out: LockedShape) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (not self.depthwise or shape_out[0] == shape_in[0])
