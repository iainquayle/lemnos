import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import identity, MergeMethod, get_features_shape, mould_features

from copy import copy, deepcopy
from math import prod


class Tree(nn.Module):
	pass

class Node(nn.Module):
	def __init__(self,  function=identity, activation=identity, batch_norm=identity, features_shape=[1], merge_method=MergeMethod.SINGLE, split_branches=[identity], return_branch=identity):
		super().__init__()
		self.function = function 
		self.activation = activation 
		self.batch_norm = batch_norm 
		self.features_shape = torch.Size(features_shape) 
		self.split_branches = split_branches 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(split_branches) > 1 else merge_method.get_function()
		self.return_branch = return_branch 
	def forward(self, x):
		x = self.mould_features(self.activation(self.batch_norm(self.function(x))))
		return self.return_branch(self.merge_function(list(map(lambda module: module(x), self.split_branches))))
	def mould_features(self, x):
		return mould_features(x, self.features_shape) 
	@staticmethod
	def new(shape_out, shape_tensor, rules=None):
		shape_in = get_features_shape(shape_tensor)
		dimensionality_in = len(shape_in)
		dimensionaltiy_out = len(shape_out)
		#can use .numel but in case of list input this is better?
		features_in = prod(list(shape_in))
		features_out = prod(list(shape_out))
		new_features = 0
		if abs(features_in - features_out) < 3: 
			new_features = features_out
		else:
			new_features = int((features_in + features_out) / 2)
		function = nn.Linear(features_in, new_features)
		batch_norm = nn.BatchNorm1d(new_features)
		features_shape = torch.Size([new_features])
		activation = nn.ReLU6()
		return_branch = identity	
		if new_features != features_out:
			return_branch = Node.new(shape_out, mould_features(function(shape_tensor), [new_features]))
		return Node(function=function, activation=activation, batch_norm=batch_norm, features_shape=features_shape, 
			return_branch=return_branch)
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
	
