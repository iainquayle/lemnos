from __future__ import annotations

import torch
from torch import Tensor, Size 
from torch.nn import Module, ModuleList
import torch.nn as nn

from src.model_structures.commons import identity, Identity, MergeMethod, get_features_shape, mould_features
from src.build_structures.priority_graphs.manual import NodePattern, Transiton, Graph as BuildGraph 

from typing import Dict, List, Set
from typing_extensions import Self
from copy import copy 
from dataclasses import dataclass

class Graph(Module):
	pass

class Node(Module):
	def __init__(self, 
			transform: Module = Identity(),
			activation: Module = nn.ReLU(), 
			batch_norm: Module = Identity(), 
			shape_in: List[int] | Size = [1], 
			merge_method: MergeMethod = MergeMethod.SINGLE, 
			node_children: List[Node] = list(),
			node_parents: List[Node] = list(),
			node_pattern: NodePattern = NodePattern()
			) -> None:
		super().__init__()
		self.transform: Module = transform 
		self.activation: Module = activation	 
		self.batch_norm: Module = batch_norm 
		self.shape_in: Size = Size(shape_in)
		self.node_children: ModuleList = ModuleList(node_children) 
		self.node_parents: ModuleList = ModuleList(node_parents) 
		self.inputs: List[Tensor] = [] 
		self.merge_function = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(node_children) > 1 else merge_method.get_function() 
		self.merge_method: MergeMethod = merge_method
		self.node_pattern: NodePattern = node_pattern 
	def forward(self, x: Tensor) -> Tensor | None:
		self.inputs.append(self.mould_input(x))
		if len(self.inputs) >= len(self.node_parents):
			x = self.activation(self.batch_norm(self.transform(self.merge_function(self.inputs))))
			y = x 
			for child in self.node_children:
				y = child(x)
			self.inputs = list() 
			return y
		else:
			return None
	def push_shape_tensor(self, x: Tensor) -> None:
		self.inputs.append(self.mould_input(x))
	def compile_flat_module_forward(self, source: str, registers: Dict[str, int]) -> str:
		return "" 
	def mould_input(self, x: Tensor) -> Tensor:
		return mould_features(x, self.shape_in) 
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


	@dataclass
	class StackData:
		node: Node 
		parents: Set[NodePattern]
		priority: int
	class Stack:
		def __init__(self, node: Node) -> None:
			self.stack: List[Node.StackData] = [Node.StackData(node, set(), 0)]
		def push(self, data: Node.StackData) -> None:
			self.stack.append(data)
		def pop(self) -> Node.StackData:
			return self.stack.pop()
		def peek(self) -> Node.StackData:
			return self.stack[-1]
		def __len__(self) -> int:
			return len(self.stack)
	@staticmethod
	def from_build_graph(build_graph: BuildGraph, shapes_in: List[Size], shape_outs: List[Size], index: int =0) -> Node:
		#general alg:
		#	pick the node with the lowest priority in the list, this will always be the top of the stack 
		#	transition out of it, use group based on constraints
		#	if there is a matching node in the stack, that hasnt been parented by the same type yet, parent it
		nodes: Dict[NodePattern, Node.Stack] = {} 
		for state in build_graph.start_states:
			nodes[state] = Node.Stack(Node())
		MAX_ITERATIONS = 1024 
		iterations = 0
		while len(nodes) > 0 and iterations < MAX_ITERATIONS:
			min_priority = Transiton.MAX_PRIORITY 
			min_node = None
			for state, stack in nodes.items():
				if stack.peek().priority < min_priority:
					min_priority = stack.peek().priority
			iterations += 1
		return Node() 

