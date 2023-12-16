from __future__ import annotations

from torch import Tensor, Size, _dim_arange
import torch.nn as nn
from torch.nn import Conv2d, Module, ModuleList

from src.model_structures.commons import Identity, MergeMethod
from src.build_structures.commons import Bound, Index

from abc import ABC as Abstract, abstractmethod 
from typing import List, Set, Dict, NamedTuple, Tuple

class NodeParameters:
	def __init__(self,
			shape_bounds: List[Bound] = [Bound()], 
			size_coefficient_bounds: Bound = Bound(),
			activation_functions: List[Module] = [Identity()], 
			merge_method: MergeMethod =MergeMethod.ADD, 
			) -> None:
		self.shape_bounds = shape_bounds 
		self.size_coefficient_bounds = size_coefficient_bounds
		self.activation_functions = activation_functions 
		self.merge_method = merge_method
	@abstractmethod
	def get_transform_string(self, shape_in: Size, index: Index =Index()) -> str:
		pass
	def get_transform(self, shape_in: Size, index: Index) -> Module:
		return eval(self.get_transform_string(shape_in, index))
	def get_activation(self, index: int) -> Module:
		return self.activation_functions[index % len(self.activation_functions)]
	def get_batch_norm(self, features: int) -> Module:
		return Identity()
	
class IdentityInfo(NodeParameters):
	def __init__(self) -> None:
		super().__init__()
#	def get_transform_string(self) -> str:
#		return "Identity()"

class BasicConvInfo(NodeParameters):
	def __init__(self,
			node_info: NodeParameters = NodeParameters(),
			dimension: int =1,
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
		out_channels = self.size_coefficient_bounds.from_ratio(index.as_ratio()) * shape_in[1]
		return f"Conv{self.dimension}d(in_channels={shape_in[1]}, out_channels={int(out_channels)}, kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation}, padding={self.padding})"
	def get_batch_norm(self, features: int) -> Module:
		if self.dimension == 1:
			return nn.BatchNorm1d(features) 
		elif self.dimension == 2:
			return nn.BatchNorm2d(features)
		elif self.dimension == 3:
			return nn.BatchNorm3d(features)
		else:
			raise ValueError("Invalid dimension")
		
#classic depth wise, where one kernel is applied to each input channel
class DepthwiseConvInfo(NodeParameters):
	pass

#one kernel, applied to each channel
class DepthwiseSharedConvInfo(NodeParameters):
	pass
