import torch
from torch import Tensor, Size 
from torch.nn import Module, ModuleList
import torch.nn as nn
import torch.nn.functional as F

from src.structures.commons import identity, Identity, MergeMethod, get_features_shape, mould_features

from typing import Callable, List, Any 
from typing_extensions import Self
from copy import copy, deepcopy
from math import prod

class Graph(Module):
	pass
class Node(Module):
	def __init__(self, 
	      function: Module =Identity(),
			activation: Module =nn.ReLU(), 
			batch_norm: Module =Identity(), 
			shape_in: List[int] | Size =[1], 
			merge_method: MergeMethod =MergeMethod.SINGLE, 
			node_children: ModuleList =ModuleList([Identity()]), 
			node_parents: ModuleList =ModuleList([])) -> None:
		super().__init__()
		self.function: Module = function 
		self.activation: Module = activation	 
		self.batch_norm: Module = batch_norm 
		self.shape_in: Size | List[int] = shape_in 
		self.node_children: ModuleList = node_children 
		self.node_parents: ModuleList = node_parents 
		self.inputs: List[Tensor] = [] 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(node_children) > 1 else merge_method.get_function() 
		self.merge_method: MergeMethod = merge_method
		#self.transition: Transition = transition 
	def forward(self, x: Tensor) -> Tensor | None:
		self.inputs.append(self.mould_input(x))
		if len(self.inputs) >= len(self.node_parents):
			x = self.activation(self.batch_norm(self.function(self.merge_function(self.inputs))))
			y = None
			for child in self.node_children:
				y = child(x)
			self.inputs = []
			return y
		else:
			return None
	@staticmethod
	def new(transition_graph, index: int):
		pass
	def push_shape_tensor(self, x: Tensor) -> None:
		self.inputs.append(self.mould_input(x))
	def compile_to_flat_module(self) -> Module:
		return Identity()
	def mould_input(self, x: Tensor) -> Tensor:
		return mould_features(x, self.shape_in) 
	def set_node_children(self, node_children: List[Module] | ModuleList) -> Self:
		self.node_children = ModuleList(node_children)
		for child in node_children:
			Node(child).node_parents.append(self)
		return self
	def add_node_child(self, node_child) -> Self:
		self.node_children.append(node_child)
		node_child.node_parents.append(self)
		return self
	def reset_inputs(self) -> None:
		#self.inputs = []
		#for child in self.node_children:
		#	child.reset_inputs()
		pass
	def check_validity(self):
		pass
	

