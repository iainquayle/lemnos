import torch
import torch.nn as nn
from copy import copy, deepcopy
from math import prod

identity = lambda x: x
CONCAT = 'concat'
ADD = 'add'
LIST = 'list'
def get_merge_function(merge_method):
	if merge_method == CONCAT:
		return lambda x: torch.cat(x, dim=1)
	elif merge_method == ADD:
		return lambda x: sum(x)
	elif merge_method == LIST:
		return identity 
	else:
		return None
def get_features_shape(x):
	return x.shape[1:]

class Tree(nn.Module):
	pass

#TODO: change so that shape does not include batch

class Node(nn.Module):
	def __init__(self, merge_method=CONCAT, function=identity, features_shape=torch.Size([0]), sub_branches=[identity], return_branch=identity):
		super().__init__()
		self.function = function 
		self.features_shape = features_shape 
		self.sub_branches = sub_branches 
		self.return_branch = return_branch 
		self.merge_function = get_merge_function(merge_method)
	def forward(self, x):
		x = self.mould_features_shape(self.function(x))
		return self.return_branch(
			self.merge_function(
			list(map(lambda module: module(x), self.sub_branches))))
	def mould_features_shape(self, x):
		return x.view([x.shape[0]] + list(self.features_shape))
	@staticmethod
	def new(shape_out, shape_tensor, model_build):
		shape_in = get_features_shape(shape_tensor)
		dimensionality_in = len(shape_in)
		dimensionltiy_out = len(shape_out)
		features_in = prod(list(shape_in))
		features_out = prod(list(shape_out))
		function = None
		features_shape = None
		if abs(features_in - features_out) < 3: 
			pass
		else:
			new_features = int((features_in + features_out) / 2)
			mould_shape = 0
			pass
		pass
	@staticmethod
	def change_shape_dimension(shape, diff=1):
		return [-1] + list(shape)[1+diff:] if diff >= 0 else [1] * -diff + list(shape)
	@staticmethod
	def squish_shape(shape, diff=1):
		return [1] * diff + [-1] + list(shape)[2:] 

print("start")			
test1 = Node(function=nn.Linear(5, 5), merge_method=CONCAT, shape=torch.Size([1, 5]), 
	sub_branches=[Node(function=nn.Linear(5, 5), merge_method=ADD, shape=torch.Size([1, 5])), Node(function=nn.Linear(5, 5), merge_method=ADD, shape=torch.Size([1, 5]))],
	return_branch=Node(function=nn.Linear(10, 10), merge_method=ADD, shape=torch.Size([1, 10])))
print(test1(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])))