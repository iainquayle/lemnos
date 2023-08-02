import torch
from torch import Tensor, Size
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

from kontrol.transitions import Transition, ConvTransition
from structures.commons import identity, Identity, MergeMethod, get_features_shape, mould_features

from typing import Callable, List, Any 
from copy import copy, deepcopy
from math import prod

class Graph(Module):
	pass
class Node(Module):
	def __init__(self, 
	      function: Module =Identity(),
			activation: Module =Identity(), 
			batch_norm: Module =Identity(), 
			#TODO: consider removing shapout
			shape_out: List[int] | Size =[1], 
			shape_in: List[int] | Size =[1], 
			merge_method: MergeMethod =MergeMethod.SINGLE, 
			children: List[Module] =[Identity()], 
			parents: List[Module] =[Identity()]) -> None:
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
		self.merge_method = merge_method
	def forward(self, x: Tensor) -> Tensor | None:
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
	def compile_to_flat_module(self) -> Module:
		pass   
	def mould_input(self, x: Tensor):
		return mould_features(x, self.shape_in) 
	def mould_output(self, x: Tensor):
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
	def check_validity(self):
		pass
	

#TODO: make sure that back pass works with the non immediate return of the forward pass

mod1 = Node()
mod2 = Node()
mod3 = Node()