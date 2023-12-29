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
	def get_output_shape(self, parent_shapes: List[Size], sibling_shapes: List[Size], index: Index =Index()) -> Size | None:
		required_shape = self.merge_method.get_required_shape(sibling_shapes)
		if required_shape is None:
			pass
		else:
			pass
		#TODO:
		#	check if there is a required shape that needs to be hit
		#		make func specific to getting a required shape 
		#	if so, make it work within the bounds
		#	else, make one using the coefficients, within the bounds
		#		make a func specific to getting merged shape
		#	these shapes must also be achievable once put through the transform
		#		fairly easy with something like a regular conv, but much harder when it will spit out some multiple of features
		#	else, none
		#TODO:
		#	consider when it makes sense, making a module specific to the verticle filter reuse
		#	and switch mergers back to requiring the same shapes
	@abstractmethod
	def get_target_output_shape(self, parent_shapes: List[Size], index: Index =Index()) -> Size:
		pass
	def get_batch_norm_string(self, features: int) -> str:
		return "Identity()"
	def get_batch_norm(self, features: int) -> Module:
		return eval(self.get_batch_norm_string(features))
	
#TODO: make tokens for constants, ie conv, bn, rbrack, comma
#or make functions, ie then can just use param list

class IdentityParameters(BaseParameters):
	def __init__(self) -> None:
		super().__init__()
	def get_transform_string(self, shape_in: Size, index: Index =Index()) -> str:
		return "Identity()"
	def get_target_output_shape(self, parent_shapes: List[Size], index: Index =Index()) -> Size:
		return self.merge_method.get_total_merged_shape(parent_shapes)

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
	def get_target_output_shape(self, parent_shapes: List[Size], index: Index =Index()) -> Size:
		pass
