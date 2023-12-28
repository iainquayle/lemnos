from __future__ import annotations

import torch
from torch import Size 
import torch.nn as nn
from torch.nn import Module, Identity

from src.build_structures.commons import Bound, Index, MergeMethod, Concat
from abc import ABC as Abstract, abstractmethod 
from typing import List, Tuple

#TODO:
#	make more unified solution for shape creation
#	will need to take into account:
#		sibling shapes
#		coefficients
#		shape bounds
#		dimensionality conversions
#NOTE: may need to change the assumtuption of always using dim 1, in the case of using higher dim convolutions for lower dim spaces
#	ie, dim 1 will always be 1, dim 2 will be effectivley channels
#	two options:
#		have an explicit dim join param
#		always join on 1, but have a secondary mould
#		another maybe option, is auto view all inputs as 2d, join on 1, then use a mould
#			should work, is still efficient, and maintains simplicity, maybe

class BaseParameters():
	def __init__(self,
			shape_bounds: List[Bound] = [Bound()], 
			size_coefficient_bounds: Bound = Bound(),
			merge_method: MergeMethod = Concat(), 
			) -> None:
		self.shape_bounds = shape_bounds 
		self.size_coefficient_bounds = size_coefficient_bounds
		self.merge_method = merge_method
	@abstractmethod
	def get_transform_string(self, shape_in: Size, index: Index) -> str:
		pass
	def get_transform(self, shape_in: Size, index: Index) -> Module:
		return eval(self.get_transform_string(shape_in, index))
	def shape_in_bounds(self, shape_in: Size) -> bool:
		for bound, dimension_size in zip(self.shape_bounds, shape_in):
			if not dimension_size in bound:
				return False
		return True 
	def get_output_shape(self, shape_in: Size, sibling_shapes: List[Size], index: Index =Index()) -> Size | None:
		output_shape: Size | None = self.get_raw_output_shape(shape_in, index)
		if self.merge_method.validate_shapes(sibling_shapes + [output_shape]): 
			return None
		return output_shape if self.shape_in_bounds(output_shape) else None
	def get_raw_output_shape(self, shape_in: Size, index: Index =Index()) -> Size:
		temp_tranform = self.get_transform(shape_in, index)
		return temp_tranform(torch.zeros(shape_in)).shape
	#change this to use the string version
	def get_batch_norm_string(self, features: int) -> str:
		return "Identity()"
	def get_batch_norm(self, features: int) -> Module:
		return eval(self.get_batch_norm_string(features))
	
class IdentityParameters(BaseParameters):
	def __init__(self) -> None:
		super().__init__()
	def get_transform_string(self, shape_in: Size, index: Index =Index()) -> str:
		return "Identity()"
	def get_raw_output_shape(self, shape_in: Size, index: Index =Index()) -> Size:
		return shape_in

class ConvParameters(BaseParameters):
	def __init__(self,
			node_info: BaseParameters = BaseParameters(),
			dimension: int = 1,
			kernel_size: Tuple | int = 1, 
			stride: Tuple | int = 1, 
			dilation: Tuple | int = 1,
			padding: Tuple | int = 1
			) -> None:
		super().__init__()
		self.__dict__.update(node_info.__dict__)
		if len(tuple([kernel_size])) == 0 or len(tuple([stride])) == 0 or len(tuple([dilation])) == 0 or len(tuple([padding])) == 0:
			exit("kernel_size, stride, dilation, padding must have at least one dimension")
		if(len(tuple([kernel_size])) > dimension or len(tuple([stride])) > dimension or len(tuple([dilation])) > dimension or len(tuple([padding])) > dimension):
			exit("kernel_size, stride, dilation, padding must match dimension")
		self.dimension = dimension
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
	def get_transform_string(self, shape_in: Size, index: Index = Index()) -> str:
		out_channels = self.size_coefficient_bounds.from_index(index, shape_in[0])
		return f"Conv{self.dimension}d(in_channels={shape_in[0]}, out_channels={int(out_channels)}, kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation}, padding={self.padding})"
	def get_batch_norm_string(self, features: int) -> str:
		return f"BatchNorm{self.dimension}d({features})"
