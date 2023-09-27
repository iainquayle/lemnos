from __future__ import annotations
from types import new_class

import torch
from torch import Tensor, Size 
from torch.nn import Module, ModuleList
import torch.nn as nn
import torch.nn.functional as F

from src.model_structures.commons import identity, Identity, MergeMethod, get_features_shape, mould_features
from src.build_structures.priority_graphs.manual import BuildNode 
from src.build_structures.module_info import ModuleInfo 

from typing import Callable, Dict, List, Any, Set, Tuple 
from typing_extensions import Self
from copy import copy, deepcopy
from math import prod

class Graph(Module):
	pass
class Node(Module):
	def __init__(self, 
	      transform: Module =Identity(),
			activation: Module =nn.ReLU(), 
			batch_norm: Module =Identity(), 
			shape_in: List[int] | Size =[1], 
			merge_method: MergeMethod =MergeMethod.SINGLE, 
			node_children: ModuleList =ModuleList([Identity()]), 
			node_parents: ModuleList =ModuleList([]),
			build_node: BuildNode = BuildNode()
			) -> None:
		super().__init__()
		self.transform: Module = transform 
		self.activation: Module = activation	 
		self.batch_norm: Module = batch_norm 
		self.shape_in: Size | List[int] = shape_in 
		self.node_children: ModuleList = node_children 
		self.node_parents: ModuleList = node_parents 
		self.inputs: List[Tensor] = [] 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(node_children) > 1 else merge_method.get_function() 
		self.merge_method: MergeMethod = merge_method
		self.build_node: BuildNode = build_node 
	def forward(self, x: Tensor) -> Tensor | None:
		self.inputs.append(self.mould_input(x))
		if len(self.inputs) >= len(self.node_parents):
			x = self.activation(self.batch_norm(self.transform(self.merge_function(self.inputs))))
			y = None
			for child in self.node_children:
				y = child(x)
			self.inputs = []
			return y
		else:
			return None
	@staticmethod
	def from_build_graph(build_graph: BuildNode, shape_in: Size, shape_out: Size, index: int =0) -> Node:
		nodes: Dict[BuildNode, List[Tuple[Node, Set[BuildNode]]]] = {} 
			
		#steps:
		#query build_graph for next transition group
		#enumerate nodes, place them in set, 
		return Node() 
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
	

