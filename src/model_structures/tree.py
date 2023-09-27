import torch
from torch import Tensor, Size
import torch.nn as nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from src.structures.commons import identity, Identity, MergeMethod, mould_features

from copy import copy, deepcopy
from math import prod
from typing import Callable, List, Any

#type ModuleFunction = Callable[[List[Tensor] | Tensor], List[Tensor] | Tensor]

class Tree(nn.Module):
	pass

class Node(nn.Module):
	def __init__(self,  
	      function: Module =Identity(), 
			activation: Module | Callable[[List[Tensor] | Tensor], List[Tensor] | Tensor] =identity, 
			batch_norm: Module =Identity(), 
			features_shape: List[int] | Size =[1], 
			merge_method: MergeMethod =MergeMethod.SINGLE, 
			split_branches: ModuleList =ModuleList([Identity()]), 
			return_branch: Module =Identity()):
		super().__init__()
		self.function = function 
		self.activation = activation 
		self.batch_norm = batch_norm 
		self.features_shape = Size(features_shape) 
		self.split_branches = split_branches 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(split_branches) > 1 else merge_method.get_function()
		self.return_branch = return_branch 
	def forward(self, x):
		x = self.mould_features(self.activation(self.batch_norm(self.function(x))))
		return self.return_branch(self.merge_function(list(map(lambda module: module(x), self.split_branches))))
	def mould_features(self, x):
		return mould_features(x, self.features_shape) 
	#@staticmethod
	#def new(index: int, shape_out: List[int] | Size, shape_tensor: List[int] | Size, transistion: Transition):
	#	function = transistion.get_function(index, shape_out, shape_tensor)
		#split = Node.new()
	#	pass
	#not sure whether to use this, may take too much explicitness out for minimal gain
	#also harder to make tree walker when loosing explicitness
	
	@staticmethod
	def get_branch_function(merge_method: MergeMethod):
		if merge_method == MergeMethod.CONCAT:
			return lambda x, split_branches, return_branch: return_branch(torch.cat(list(map(lambda module: module(x), split_branches)), dim=1))
		elif merge_method == MergeMethod.ADD:
			return lambda x, split_branches, return_branch: return_branch(sum(list(map(lambda module: module(x), split_branches))))
		elif merge_method == MergeMethod.SINGLE:
			return lambda x, split_branches, return_branch: return_branch(split_branches[0](x))
		elif merge_method == MergeMethod.LIST:
			return lambda x, split_branches, return_branch: list(map(lambda module: module(x), split_branches))
		else:
			return None
	