from __future__ import annotations

from torch import Tensor, Size, _dim_arange
from torch.nn import Conv2d, Module, ModuleList
from src.structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod 
from typing import List, Set, Dict, NamedTuple, Tuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

class Bound:
	def __init__(self, lower: int | float =1, upper: int | float =1) -> None:
		if upper < lower:
			exit("upper smaller than lower bound")
		self.upper: int | float = upper
		self.lower: int | float = lower 
	def difference(self) -> int | float:
		return self.upper - self.lower
	def average(self) -> int | float:
		return (self.upper + self.lower) / 2
	def inside(self, i: int | float) -> bool:
		return i >= self.lower and i <= self.upper	
	def from_ratio(self, ratio: float) -> float:
		return self.lower + ratio * self.difference()
class Index:
	MAX_INDEX = 2**16 -1
	def __init__(self, index: int =0) -> None:
		self.set_index(index)
	def set_index(self, index: int) -> None:
		self.index = index % Index.MAX_INDEX
	def as_ratio(self) -> float:
		return self.index / Index.MAX_INDEX


#TODO: consider making an optional argument for the base, takes itself and updates the dict
#thus automatically dealing with super init
class NodeInfo:
	def __init__(self,
	      shape_bounds: List[Bound] =[Bound()], 
			size_coefficient_bounds: Bound =Bound(),
			activation_functions: List[Module] =[Identity()], 
			merge_method: MergeMethod =MergeMethod.ADD, 
			use_batch_norm: bool =True) -> None:
		super().__init__()
		self.shape_bounds = shape_bounds 
		self.size_coefficient_bounds = size_coefficient_bounds
		self.activation_functions = activation_functions 
		self.merge_method = merge_method
		self.use_batch_norm = use_batch_norm 
	@abstractmethod
	def get_function_string(self, shape_in: Size, index: Index =Index()) -> str:
		pass
	def get_function(self, shape_in: Size, index: Index) -> Module:
		return eval(self.get_function_string(shape_in, index))
	
class IdentityInfo(NodeInfo):
	def __init__(self) -> None:
		super().__init__()
	def get_function_string(self) -> str:
		return "Identity()"

#TODO: maybe super conv, and subs, because shape needs to change, not based on input
class BasicConvInfo(NodeInfo):
	def __init__(self,
		node_info: NodeInfo =NodeInfo(),
		kernel_size: Tuple | int = 1, 
		stride: Tuple | int = 1, 
		dilation: Tuple | int = 1,
		padding: Tuple | int = 1) -> None:
		super().__init__()
		self.__dict__.update(node_info.__dict__)
		if len(tuple([kernel_size])) == 0 or len(tuple([stride])) == 0 or len(tuple([dilation])) == 0 or len(tuple([padding])) == 0:
			exit("kernel_size, stride, dilation, padding must have at least one dimension")
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.padding = padding
	def get_function_string(self, shape_in: Size, index: Index) -> str:
		dimensions = len(shape_in) - 2
		out_channels = self.size_coefficient_bounds.from_ratio(index.as_ratio()) * shape_in[1]
		return f"Conv{dimensions}d(in_channels={shape_in[1]}, out_channels={int(out_channels)}, kernel_size={self.kernel_size}, stride={self.stride}, dilation={self.dilation}, padding={self.padding})"
    
class DepthwiseConvInfo(NodeInfo):
	pass

class DepthwiseSharedConvInfo(NodeInfo):
	pass