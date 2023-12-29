from __future__ import annotations

import torch
from torch import Size 
import torch.nn as nn
from torch.nn import Module, Identity

from src.build_structures.commons import Bound, Index, MergeMethod, Concat
from abc import ABC as Abstract, abstractmethod 
from typing import List, Tuple

class BaseParameters():
	def __init__(self,
			shape_bounds: List[Bound] = [Bound()], 
			size_coefficient_bounds: Bound = Bound(),
			merge_method: MergeMethod = Concat(), 
			) -> None:
		self.shape_bounds = shape_bounds 
		self.size_coefficient_bounds = size_coefficient_bounds
		self.merge_method = merge_method
	def shape_in_bounds(self, shape_in: Size) -> bool:
		for bound, dimension_size in zip(self.shape_bounds, shape_in):
			if not dimension_size in bound:
				return False
		return True 
	@abstractmethod
	def validate_output_shape(self, shape_out: Size) -> bool:
		pass
	def get_output_shape(self, parent_shapes: List[Size], sibling_shapes: List[Size], index: Index =Index()) -> Size | None:
		required_size = self.merge_method.get_required_size(sibling_shapes)
		if required_size is None:
			pass
		else:
			pass
		#TODO:
		#	check if there is a required shape that needs to be hit
		#	else, make one using the coefficients, within the bounds
		#	these shapes must also be achievable once put through the transform
		#		fairly easy with something like a regular conv, but much harder when it will spit out some multiple of features
		#	else, none
	def get_mould_shape(self, parent_shapes: List[Size]) -> Size:
		if len(parent_shapes) == 0:
			return Size([int(bound.lower) for bound in self.shape_bounds])
		else:
			max_dim_shape = Size()
			for parent_shape in parent_shapes:
				if len(parent_shape) > len(max_dim_shape):
					max_dim_shape = parent_shape
			total_merged_size: int = self.merge_method.get_total_merged_size(parent_shapes)
			shape_list: List[int] = list(max_dim_shape[max(1, len(max_dim_shape) - len(self.shape_bounds) + 1):])
			for dimension_size in shape_list:
				if total_merged_size % dimension_size != 0:
					raise Exception("wtf mould shape failed")
				total_merged_size = int(total_merged_size / dimension_size)
			shape_list = ([1] * (len(self.shape_bounds) - len(shape_list) - 1)) + [total_merged_size] + shape_list
			return Size(shape_list)
	@abstractmethod
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str | None:
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
	def __init__(self) -> None:
		super().__init__()
	def get_closest_shape(self, target_shape: Size, parent_shapes: List[Size]) -> Size:
		return self.get_mould_shape(parent_shapes)
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str | None:
		return "Identity()"

def auto_fill_tuple(val: Tuple | int, bounds: List[Bound]) -> Tuple:
	return val if isinstance(val, tuple) else tuple([val] * (len(bounds) - 1))
class ConvParameters(BaseParameters):
	def __init__(self,
			node_info: BaseParameters = BaseParameters(),
			kernel_size: Tuple | int = 1, 
			stride: Tuple | int = 1, 
			dilation: Tuple | int = 1,
			padding: Tuple | int = 1
			) -> None:
		super().__init__()
		self.__dict__.update(node_info.__dict__)
		if len(tuple([kernel_size])) == 0 or len(tuple([stride])) == 0 or len(tuple([dilation])) == 0 or len(tuple([padding])) == 0:
			exit("kernel_size, stride, dilation, padding must have at least one dimension")
		self.kernel_size: Tuple = auto_fill_tuple(kernel_size, self.shape_bounds)
		self.stride: Tuple = auto_fill_tuple(stride, self.shape_bounds)
		self.dilation: Tuple = auto_fill_tuple(dilation, self.shape_bounds)
		self.padding: Tuple = auto_fill_tuple(padding,  self.shape_bounds)
	def validate_output_shape(self, shape_out: Size) -> bool:
		i = 1
		while (i < len(shape_out) 
				and (shape_out[i] - (int(shape_out[i] / self.kernel_size[i]) + self.padding[i] * 2)) % self.stride[i] == 0
				and shape_out[i] in self.shape_bounds[i]):
			i += 1
		return i == len(shape_out) and shape_out[0] in self.shape_bounds[0]
	def get_transform_src(self, shape_in: Size, shape_out: Size) -> str | None:
		i = 1
		while i < len(shape_out) and (shape_out[i] - (int(shape_out[i] / self.kernel_size[i]) + self.padding[i] * 2)) % self.stride[i] == 0:
			i += 1
		if i == len(shape_in):
			return ""
		else:
			return None
	def get_batch_norm_src(self, shape_out: Size) -> str:
		return ""
