import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
from math import prod
from enum import Enum

identity = lambda x: x
class MergeMethod(Enum):
	CONCAT = 'concat'
	ADD = 'add'
	SINGLE = 'single'
	LIST = 'list'
	def get_function(self):
		if self == MergeMethod.CONCAT:
			return lambda x: torch.cat(x, dim=1)
		elif self == MergeMethod.ADD:
			return lambda x: sum(x)
		elif self == MergeMethod.SINGLE:
			return lambda x: x[0]
		elif self == MergeMethod.LIST:
			return identity 
		else:
			return None
def get_features_shape(x):
	return x.shape[1:]
def get_batch_norm(shape):
	if len(shape) == 1:
		return nn.BatchNorm1d(shape.numel())
	elif len(shape) == 2:
		return nn.BatchNorm2d(shape.numel())
def mould_features(x, shape):
    return x.view([x.shape[0]] + list(shape))
@staticmethod
def change_shape_dimension(shape, diff=1):
	return [-1] + list(shape)[1+diff:] if diff >= 0 else [1] * -diff + list(shape)
@staticmethod
def squish_shape(shape, diff=1):
	return [1] * diff + [-1] + list(shape)[2:] 

class Tree(nn.Module):
	pass

class Node(nn.Module):
	def __init__(self,  function=identity, activation=identity, batch_norm=identity, features_shape=[1], merge_method=MergeMethod.SINGLE, sub_branches=[identity], return_branch=identity):
		super().__init__()
		self.function = function 
		self.activation = activation 
		self.batch_norm = batch_norm 
		self.features_shape = torch.Size(features_shape) 
		self.sub_branches = sub_branches 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(sub_branches) > 1 else merge_method.get_function()
		self.return_branch = return_branch 
	def forward(self, x):
		x = self.mould_features(self.activation(self.batch_norm(self.function(x))))
		return self.return_branch(self.merge_function(list(map(lambda module: module(x), self.sub_branches))))
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
	
class Rules:
	CONV2D = 'conv2d'
	pass


print("start")			
tens = torch.rand((2, 10), dtype=torch.float32)
test = Node.new([5], tens)
print(test(tens))