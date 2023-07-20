import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import identity, MergeMethod, get_features_shape, mould_features

from copy import copy, deepcopy
from math import prod

class Graph(nn.Module):
	pass

#TODO consider removing parents refernces
class Node(nn.Module):
	def __init__(self, function=identity, activation=identity, batch_norm=identity, shape_out=[1], shape_in=[1], merge_method=MergeMethod.SINGLE, children=[identity], parents=[identity]):
		super().__init__()
		self.function = function 
		self.activation = activation	 
		self.batch_norm = batch_norm 
		self.shape_out = shape_out 
		self.shape_in = shape_in 
		self.children = children 
		self.parents = parents 
		self.inputs = [] 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(children) > 1 else merge_method.get_function() 
	def forward(self, x):
		self.inputs.append(self.mould_input(x))
		if len(self.inputs) >= len(self.parents):
			x = self.mould_output(self.activation(self.batch_norm(self.function(self.merge_function(self.inputs)))))
			y = None
			for child in self.children:
				y = child(x)
			self.inputs = []
			return y
		else:
			return None
	def mould_input(self, x):
		return mould_features(x, self.shape_in) 
	def mould_output(self, x):
		return mould_features(x, self.shape_out) 
	def add_child(self, child):
		self.children.append(child)
		child.parents.append(self)
		return self
	def add_parent(self, parent):
		self.parents.append(parent)
		parent.children.append(self)
		return self
	def reset_inputs(self):
		self.inputs = []
		for child in self.children:
			child.reset_inputs()