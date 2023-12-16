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
			mould_shape: List[int] | Size = [], 
			merge_method: MergeMethod = MergeMethod.SINGLE, 
			node_children: List[Self] = list(),
			node_parents: List[Self] = list(),
			node_pattern: NodePattern = NodePattern()
			) -> None:
		super().__init__()
		self.transform: Module = transform 
		self.activation: Module = activation	 
		self.batch_norm: Module = batch_norm 
		self.mould_shape: Size = Size(mould_shape)
		self.node_children: ModuleList = ModuleList(node_children) 
		self.node_parents: ModuleList = ModuleList(node_parents) 
		self.inputs: List[Tensor] = [] 
		self.merge = MergeMethod.CONCAT.get_function() if merge_method == MergeMethod.SINGLE and len(node_children) > 1 else merge_method.get_function() 
		self.merge_method: MergeMethod = merge_method
		self.node_pattern: NodePattern = node_pattern 
		self.output_shape: Size = Size([])
	def forward(self, x: Tensor) -> Tensor | None:
		self.inputs.append(self.mould(x))
		if len(self.inputs) >= len(self.node_parents):
			x = self.activation(self.batch_norm(self.transform(self.merge(self.inputs))))
			y = x 
			for child in self.node_children:
				y = child(x)
			self.inputs = list() 
			return y
		else:
			return None
	def mould(self, x: Tensor) -> Tensor:
		return mould_features(x, self.mould_shape)
	def compile_flat_module_forward(self, source: str, registers: Dict[str, int]) -> str:
		#will need to make some meta objects to keep track of the registers, and whether all childten have been satsified
		return "" 
	def set_node_children(self, node_children: List[Self]) -> Self:
		self.node_children = ModuleList(node_children)
		for child in node_children:
			child.node_parents.append(self)
		return self
	def add_node_child(self, node_child: Self) -> Self:
		self.node_children.append(node_child)
		node_child.node_parents.append(self)
		return self
	#process for node creation:
	#	init
	#		give pattern, first parent, activation, and merge method
	#			consider leaving all of these until the build, so that nothing is missed
	#			could even not init a node until then, and just track the parents
	#	build(duirng expand)
	#		build shape based on parents and pattern
	#			abviously needs to take into accound parents, however, the exact conversion constraints in the pattern may need to be flushed out
	#		from shape, build transform and batch norm
	@staticmethod
	def build(node_pattern: NodePattern, node_data: Node.StackData, index: int) -> None:
		node = Node(node_pattern=node_pattern,
				activation=node_pattern.node_parameters.get_activation(index),
				merge_method=node_pattern.node_parameters.merge_method)
		#OOPS: need to put node back in the stack, it is how the true parents are tracked rather than just the patterns
		output_shape_tensor = None
		for node_parent in node_data.parents:
			output_shape_tensor = self(Tensor.new_zeros(get_features_shape(node_parent.output_shape)))
			pass
		self.zero_grad()
		if output_shape_tensor is None:
			raise Exception("wtf")
		else:
			self.output_shape = Size(output_shape_tensor.shape)

	@dataclass
	class StackData:
		parents: Set[NodePattern]
		priority: int
	class Stack:
		def __init__(self, node: Node) -> None:
			self.stack: List[Node.StackData] = [Node.StackData(set(), 0)]
		def push(self, data: Node.StackData) -> None:
			self.stack.append(data)
		def pop(self) -> Node.StackData:
			return self.stack.pop()
		def peek(self) -> Node.StackData:
			return self.stack[-1]
		def __len__(self) -> int:
			return len(self.stack)
	#figure out:
	#	how to mutate the graph
	#	two plauseable options:
	#		make the indexing of the graphs some what continuous, and somehow make larger changes based on index similarity
	#			likely not the greatest
	#		choose some arbitrary point in the graph to cut, and rebuild
	#			to do so however, would require a deconstruction of the original graph so that nodes that require a certain child can get it
	#			deconstruction would need to start on nodes that have no children, work backwards, anything on the stacks that dont have parents may be removed
	#			all of the remaining nodes in the stack would be the seed for the new graph
	@staticmethod
	def from_build_graph(build_graph: BuildGraph, shapes_in: List[Size], shape_outs: List[Size], index: int =0) -> Node:
		#general alg:
		#	pick the node with the lowest priority in the list, this will always be the top of the stack 
		#	transition out of it, use group based on constraints
		#	if there is a matching node in the stack, that hasnt been parented by the same type yet, parent it
		#todo
		#	build nodes from parameters
		#	make decision on transition group based on current model structure
		input_nodes: List[Node] = []
		nodes: Dict[NodePattern, Node.Stack] = {} 
		for pattern in build_graph.start_patterns:
			node = Node()
			input_nodes.append(node)
			nodes[pattern] = Node.Stack(node)
		MAX_ITERATIONS = 1024 
		iterations = 0
		while len(nodes) > 0 and iterations < MAX_ITERATIONS:
			min_node_pattern = None 
			min_priority = Transiton.MAX_PRIORITY + 1
			for pattern, stack in nodes.items():
				if stack.peek().priority < min_priority:
					min_priority = stack.peek().priority
					min_node_pattern = pattern 
			if min_node_pattern is None:
				raise Exception("broken wtf")
			else:
				node_data = nodes[min_node_pattern].pop()
				for transition in min_node_pattern.transitions[0].transitions:
					if transition.next_pattern in nodes:
						i = len(nodes[transition.next_pattern])
						found = False
						while i > 0 and not found:
							if transition.next_pattern not in nodes[transition.next_pattern].stack[i].parents:
								nodes[transition.next_pattern].stack[i].parents.add(transition.next_pattern)
								nodes[transition.next_pattern].stack[i].priority = transition.priority
								found = True
							i -= 1
					else:
						nodes[transition.next_pattern] = Node.Stack(Node())
			iterations += 1
		return Node() 

