from __future__ import annotations

import torch
from torch import Tensor, Size 
from torch.nn import Module, ModuleList
import torch.nn as nn

from src.model_structures.commons import identity, Identity, MergeMethod, get_features_shape, mould_features
from src.build_structures.priority_graphs.manual import State 
from src.build_structures.module_info import ModuleInfo 

from typing import Callable, Dict, List, Any, Set, Tuple 
from typing_extensions import Self
from copy import copy 

class Graph(Module):
	pass
class Node(Module):
	def __init__(self, 
			transform: Module =Identity(),
			activation: Module =nn.ReLU(), 
			batch_norm: Module =Identity(), 
			shape_in: List[int] | Size =[1], 
			merge_method: MergeMethod =MergeMethod.SINGLE, 
			node_children: ModuleList =ModuleList([]), 
			node_parents: ModuleList =ModuleList([]),
			build_node: State = State()
			) -> None:
		super().__init__()
		self.transform: Module = transform 
		self.activation: Module = activation	 
		self.batch_norm: Module = batch_norm 
		self.shape_in: Size = Size(shape_in)
		self.node_children: ModuleList = node_children 
		self.node_parents: ModuleList = node_parents 
		self.inputs: List[Tensor] = [] 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(node_children) > 1 else merge_method.get_function() 
		self.merge_method: MergeMethod = merge_method
		self.build_node: State = build_node 
	def forward(self, x: Tensor) -> Tensor | None:
		self.inputs.append(self.mould_input(x))
		if len(self.inputs) >= len(self.node_parents):
			x = self.activation(self.batch_norm(self.transform(self.merge_function(self.inputs))))
			#if there are no children, return tensor directly
			y = x 
			for child in self.node_children:
				y = child(x)
			self.inputs = list() 
			return y
		else:
			return None
	@staticmethod
	def from_build_graph(build_graph: State, shapes_in: List[Size], shape_outs: List[Size], index: int =0) -> Node:
		#steps:
		#query build_graph for next transition group
		#enumerate nodes, place them in set, 
		nodes: Dict[State, List[Tuple[Node, Set[State]]]] = {} 
		while False:
			pass
		return Node() 
	def push_shape_tensor(self, x: Tensor) -> None:
		self.inputs.append(self.mould_input(x))
	def compile_flat_module_forward(self, source: str, registers: Dict[str, int]) -> str:
		return "" 
	def mould_input(self, x: Tensor) -> Tensor:
		return mould_features(x, self.shape_in) 
	#TODO: does this need to be changed? how is the value returned?
	def set_node_children(self, node_children: List[Node]) -> Self:
		self.node_children = ModuleList(node_children)
		for child in node_children:
			child.node_parents.append(self)
		return self
	def add_node_child(self, node_child: Node) -> Self:
		self.node_children.append(node_child)
		node_child.node_parents.append(self)
		return self
	def reset_inputs(self) -> None:
		#self.inputs = []
		#for child in self.node_children:
		#	child.reset_inputs()
		pass
	

