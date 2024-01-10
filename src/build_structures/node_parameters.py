from __future__ import annotations

import torch
from torch import Size 
import torch.nn as nn
from torch.nn import Module, Identity

from src.build_structures.commons import ConformanceShape, Bound, Range, Index, MergeMethod, Concat
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
#TODO: consider making it such that a shape of lower dim cant be used in a higher dim
class BaseParameters():
	def __init__(self) -> None:
		self.shape_bounds = Bound() 
		self.merge_method = Concat() 
	def validate_output_shape(self, shape_in: Size, shape_out: Size) -> bool:
		return self.validate_output_shape_transform(shape_in, shape_out) and shape_out in self.shape_bounds
	@abstractmethod
	def validate_output_shape_transform(self, shape_in: Size, shape_out: Size) -> bool:
		pass
	def get_conformance_shape(self, sibling_shapes: List[Size]) -> ConformanceShape:
		return self.merge_method.get_conformance_shape(sibling_shapes, self.shape_bounds)
	def get_mould_and_output_shapes(self, parent_shapes: List[Size], conformance_shape: ConformanceShape, index: Index = Index()) -> Tuple[Size, Size] | None:
		mould_shape = self.get_mould_shape(parent_shapes)
		output_shape = self.get_output_shape(mould_shape, conformance_shape, index)
		if output_shape is None:
			return None
		else:
			return mould_shape, output_shape
	@abstractmethod
	def get_output_shape(self, input_shape: Size, conformance_shape: ConformanceShape, index: Index = Index()) -> Size | None:
		pass
	def get_mould_shape(self, parent_shapes: List[Size]) -> Size:
		if len(parent_shapes) == 0:
			raise Exception("cannot get mould shape from empty parent shapes")
		else:
			max_dim_shape = Size()
			for parent_shape in parent_shapes:
				if len(parent_shape) > len(max_dim_shape):
					max_dim_shape = parent_shape
			total_merged_size: int = self.merge_method.get_total_merged_size(parent_shapes)
			mould_list = list(max_dim_shape[max(1, len(max_dim_shape) - len(self.shape_bounds) + 1):])
			mould_list = [total_merged_size // prod(mould_list)] + mould_list
			mould_list = ([1] * (len(self.shape_bounds) - len(mould_list))) + mould_list
			return Size(mould_list)
	@abstractmethod
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str:
		pass
	def get_transform(self, shape_in: Size, shape_out: Size) -> Module:
		return Identity() #eval(self.get_transform_src(shape_in, shape_out))
	@abstractmethod
	def get_batch_norm_src(self, shape_out: Size) -> str:
		pass
	def get_batch_norm(self, shape_out: Size) -> Module:
		return eval(self.get_batch_norm_src(shape_out))
	
class IdentityParameters(BaseParameters):
	def __init__(self, shape_bounds: Bound = Bound(), merge_method: MergeMethod = Concat()) -> None:
		self.shape_bounds = shape_bounds
		self.merge_method = merge_method
	def validate_output_shape_transform(self, shape_in: Size, shape_out: Size) -> bool:
		return shape_in == shape_out
	def get_output_shape(self, input_shape: Size, conformance_shape: ConformanceShape, index: Index = Index()) -> Size | None:
		min_dim = min(len(input_shape), len(conformance_shape.partial_shape))
		for i in range(min_dim):
			if input_shape[i] != conformance_shape.partial_shape[i]:
				return None
		#TODO: get the rest of the check down
		return input_shape
	def get_output_shape_transform(self, input_shape: Size, required_size: int | None, index: Index = Index()) -> Size | None:
		return input_shape if required_size is None or required_size == prod(input_shape) else None
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str:
		return "Identity()"

#TODO: move to commons
def auto_fill_tuple(val: Tuple | int, bounds: Bound) -> Tuple:
	return val if isinstance(val, tuple) else tuple([val] * (len(bounds) - 1))
class ConvParameters(BaseParameters):
	def __init__(self,
			shape_bounds: Bound = Bound(),
			merge_method: MergeMethod = Concat(),
			kernel: Tuple | int = 1, 
			stride: Tuple | int = 1, 
			dilation: Tuple | int = 1,
			padding: Tuple | int = 1,
			depthwise: bool = False,
			) -> None:
		if len(shape_bounds) < 2:
			exit("shape_bounds must have at least two dimensions")
		self.shape_bounds = shape_bounds
		self.size_coefficents = Range()
		self.merge_method = merge_method
		self.kernel: Tuple = auto_fill_tuple(kernel, shape_bounds)
		self.stride: Tuple = auto_fill_tuple(stride, shape_bounds)
		self.dilation: Tuple = auto_fill_tuple(dilation, shape_bounds)
		self.padding: Tuple = auto_fill_tuple(padding, shape_bounds)
		if (len(self.kernel) != len(self.stride) 
		  		or len(self.stride) != len(self.dilation) 
		  		or len(self.dilation) != len(self.padding) 
		  		or len(self.padding) != len(self.shape_bounds) - 1):
			exit("kernel, stride, dilation, padding must all have the same length and be one less than shape_bounds")
		self.depthwise: bool = depthwise
	def output_dim_to_input_dim(self, output_shape: Size, i: int) -> int:
		shape_i = i
		i -= 1
		return output_shape[shape_i] * self.padding[i] + (self.kernel[i] - 1) - (self.stride[i] - 1) - (2 * self.padding[i])
	def input_dim_to_output_dim(self, input_shape: Size, i: int) -> int:
		shape_i = i
		i -= 1
		return (input_shape[shape_i] + (2 * self.padding[i]) - (self.kernel[i] - 1) + (self.stride[i] - 1)) // self.stride[i]
	def validate_output_shape_transform(self, shape_in: Size, shape_out: Size) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (not self.depthwise or shape_out[0] == shape_in[0])
	def get_output_shape_transform(self, input_shape: Size, required_size: int | None, index: Index = Index()) -> Size | None:
		initial_shape = [self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))]
		if required_size is None:
			pass
		else:
			return size_to_shape(required_size, Size(initial_shape)) 
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str | None:
		return ""
	def get_batch_norm_src(self, shape_out: Size) -> str:
		return ""
