import torch
from torch import Tensor, Size
import torch.nn as nn
import torch.nn.functional as F

from kontrol.transitions import Transition, ConvTransition 

from structures.commons import identity, Indentity, MergeMethod, mould_features

from copy import copy, deepcopy
from math import prod
from typing import Callable, List, Any

#type ModuleFunction = Callable[[List[Tensor] | Tensor], List[Tensor] | Tensor]

class Tree(nn.Module):
	pass

class Node(nn.Module):
	def __init__(self,  
	      module_function: nn.Module =identity, 
			activation: nn.Module | Callable[[List[Tensor] | Tensor], List[Tensor] | Tensor] =identity, 
			batch_norm: nn.Module =identity, 
			features_shape: List[int] | Size =[1], 
			merge_method: MergeMethod =MergeMethod.SINGLE, 
			split_branches: List[nn.Module] =[identity], 
			return_branch: nn.Module =identity):
		super().__init__()
		self.module_function = module_function 
		self.activation = activation 
		self.batch_norm = batch_norm 
		self.features_shape = Size(features_shape) 
		self.split_branches = split_branches 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(split_branches) > 1 else merge_method.get_function()
		self.return_branch = return_branch 
	def forward(self, x):
		x = self.mould_features(self.activation(self.batch_norm(self.module_function(x))))
		return self.return_branch(self.merge_function(list(map(lambda module: module(x), self.split_branches))))
	def mould_features(self, x):
		return mould_features(x, self.features_shape) 
	@staticmethod
	def new(index: int, shape_out: List[int] | Size, shape_tensor: List[int] | Size, transistion: Transition):
		function = transistion.get_function(index, shape_out, shape_tensor)
		#split = Node.new()
		return None	
	#probably will never use this one specifically, though the general idea is maybe good
	#takes away too much flexibility, for example, when wanting a module with a res that branches but merges prior to the residual
	@staticmethod
	def get_branch_function(has_residual: bool, has_concat: bool, returns_list: bool):
		if returns_list:
			return lambda x, split_branches, return_branch: list(map(lambda module: module(x), split_branches))
		elif has_concat and has_residual:
			return lambda x, split_branches, return_branch: return_branch(x + torch.cat(list(map(lambda module: module(x), split_branches))))
		elif has_concat:
			return lambda x, split_branches, return_branch: return_branch(torch.cat(list(map(lambda module: module(x), split_branches))))
		elif has_residual:
			return lambda x, split_branches, return_branch: return_branch(x + split_branches[0](x))
		elif not has_concat and not has_residual and not returns_list:
			return lambda x, split_branches, return_branch: return_branch(x)
		else:
			print("error in get_branch_function")
			exit(1)
	