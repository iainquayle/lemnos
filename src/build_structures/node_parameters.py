from __future__ import annotations

import torch
from torch import Size 
import torch.nn as nn
from torch.nn import Module, Identity

from src.build_structures.commons import size_to_shape, Bound, Range, Index, MergeMethod, Concat
from abc import ABC as Abstract, abstractmethod 
from typing import List, Tuple

from math import prod

class BaseParameters():
	def __init__(self) -> None:
		self.shape_bounds = Bound() 
		self.merge_method = Concat() 
	def validate_output_shape(self, shape_in: Size, shape_out: Size) -> bool:
		return self.validate_output_shape_sub(shape_in, shape_out) and shape_out in self.shape_bounds
	@abstractmethod
	def validate_output_shape_sub(self, shape_in: Size, shape_out: Size) -> bool:
		pass
	def get_mould_and_output_shape(self, parent_shapes: List[Size], required_sizes: List[int], index: Index = Index()) -> Tuple[Size, Size] | None:
		mould_shape = self.get_mould_shape(parent_shapes)
		required_size = None
		for size in required_sizes:
			if size is not None and size != required_sizes[0]:
				return None
			else:
				required_size = size 
		output_shape = self.get_output_shape_sub(mould_shape, required_size, index)
		return None if output_shape == None or not self.validate_output_shape(mould_shape, output_shape) else (mould_shape, output_shape)
	#could make this not return error
	@abstractmethod
	def get_output_shape_sub(self, input_shape: Size, required_size: int | None, index: Index = Index()) -> Size | None:
		pass
	def get_mould_shape(self, parent_shapes: List[Size]) -> Size:
		if len(parent_shapes) == 0:
			return Size([int(bound) for bound in self.shape_bounds.lower])
		else:
			max_dim_shape = Size()
			for parent_shape in parent_shapes:
				if len(parent_shape) > len(max_dim_shape):
					max_dim_shape = parent_shape
			total_merged_size: int = self.merge_method.get_total_merged_size(parent_shapes)
			mould_shape = size_to_shape(total_merged_size, max_dim_shape[max(1, len(max_dim_shape) - len(self.shape_bounds) + 1):])
			if mould_shape is None:
				raise Exception("wtf mould shape failed")
			shape_list = ([1] * (len(self.shape_bounds) - len(mould_shape))) + list(mould_shape)
			return Size(shape_list)
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
	
#TODO: make tokens for constants, ie conv, bn, rbrack, comma
#or make functions, ie then can just use param list

class IdentityParameters(BaseParameters):
	def __init__(self, shape_bounds: Bound = Bound(), merge_method: MergeMethod = Concat()) -> None:
		super().__init__()
		self.shape_bounds = shape_bounds
		self.merge_method = merge_method
	def validate_output_shape_sub(self, shape_in: Size, shape_out: Size) -> bool:
		return shape_in == shape_out
	def get_output_shape_sub(self, input_shape: Size, required_size: int | None, index: Index = Index()) -> Size | None:
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
		super().__init__()
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
	def validate_output_shape_sub(self, shape_in: Size, shape_out: Size) -> bool:
		i = 1
		while i < len(shape_out) and self.output_dim_to_input_dim(shape_out, i) == shape_in[i]:
			i += 1
		return i == len(shape_out) and (not self.depthwise or shape_out[0] == shape_in[0])
	def get_output_shape_sub(self, input_shape: Size, required_size: int | None, index: Index = Index()) -> Size | None:
		initial_shape = [self.input_dim_to_output_dim(input_shape, i) for i in range(1, len(input_shape))]
		if required_size is None:
			pass
		else:
			return size_to_shape(required_size, Size(initial_shape)) 
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str | None:
		return ""
	def get_batch_norm_src(self, shape_out: Size) -> str:
		return ""
