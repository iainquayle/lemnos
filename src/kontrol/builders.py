from __future__ import annotations

from torch import Tensor, Size
from torch.nn import Module, ModuleList
from src.structures.commons import Identity, MergeMethod
from abc import ABC, abstractmethod 
from typing import List, Set, Dict, NamedTuple
from typing_extensions import Self
from collections import namedtuple
import gc
from copy import copy

class Bound:
	def __init__(self, lower: int | float =1, upper: int | float =1) -> None:
		if upper < lower:
			exit("upper smaller than lower bound")
		if upper < 0 or lower < 0:
			exit("bound in negative")
		self.upper: int | float = upper
		self.lower: int | float = lower 
	def average(self) -> int | float:
		return (self.upper + self.lower) / 2
	def inside(self, i: int | float) -> bool:
		return i >= self.lower and i <= self.upper	

class Builder:
	def __init__(self,
	      shape_bounds: List[Bound] =[Bound()], 
			shape_coefficient_bounds: List[Bound] =[Bound()],
			activation_functions: List[Module] =[Identity()], 
			merge_method: MergeMethod =MergeMethod.ADD, 
			use_batch_norm: bool =True) -> None:
		super().__init__()
		self.shape_bounds = shape_bounds 
		self.shape_coefficient_bounds = shape_coefficient_bounds
		self.activation_functions = activation_functions 
		self.merge_method = merge_method
		self.use_batch_norm = use_batch_norm 
	@abstractmethod
	def build_function(self, input: Tensor) -> Module:
		pass
	
class IdentityBuilder(Builder):
	def __init__(self) -> None:
		super().__init__()
	def build_function(self, input: Tensor, index: int) -> Module:
		return Identity()
	
class ConvBuilder(Builder):
	def __init__(self,
		kernel_sizes: List[Bound] = [Bound()],
		stride: Bound = Bound(), 
		dilation: Bound = Bound(),
		padding: Bound = Bound()) -> None:
		super().__init__()
	def build_function(self, input: Tensor, index: int) -> Module:
		return Identity()	