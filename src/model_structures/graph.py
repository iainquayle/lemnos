from __future__ import annotations

import torch
from torch import Tensor, Size 
from torch.nn import Module, ModuleList
import torch.nn as nn

from src.model_structures.commons import identity, Identity, MergeMethod, get_features_shape, mould_features
from src.build_structures.priority_graphs.manual import NodePattern, MAX_PRIORITY, Graph as BuildGraph 

from typing import Dict, List, Set
from typing_extensions import Self
from copy import copy 
from dataclasses import dataclass

#TODO: need to implement some means by which a node can check what dims a node may already have that it is connecting to
#TODO misc: need to make it such that the mould shape always infers the 1th dimension

class Graph(Module):
	def __init__(self):
		super().__init__()
		self.input_nodes: ModuleList = ModuleList()
		self.output_nodes: ModuleList = ModuleList()
		self.input_patterns: List[NodePattern] = list()
		self.input_shapes: List[Size] = list()
	#TODO: find better name for this
	@dataclass
	class StackData:
		parents: Dict[NodePattern, Node]
		priority: int
	class Stack:
		def __init__(self) -> None:
			self.stack: List[Graph.StackData] = [Graph.StackData(dict(), 0)]
		def push(self, data: Graph.StackData) -> None:
			self.stack.append(data)
		def pop(self) -> Graph.StackData:
			return self.stack.pop()
		def peek(self) -> Graph.StackData:
			return self.stack[-1]
		def __len__(self) -> int:
			return len(self.stack)
	@staticmethod
	def from_build_graph(build_graph: BuildGraph, shapes_in: List[Size], shape_outs: List[Size]) -> Graph:
		input_nodes: List[Node] = []
		nodes: Dict[NodePattern, Graph.Stack] = {} 
		for pattern in build_graph.start_patterns:
			node = Node()
			input_nodes.append(node)
			nodes[pattern] = Graph.Stack()
		MAX_ITERATIONS = 1024 
		iterations = 0
		while len(nodes) > 0 and iterations < MAX_ITERATIONS:
			min_node_pattern = None 
			min_priority = MAX_PRIORITY + 1
			for pattern, stack in nodes.items():
				if stack.peek().priority < min_priority:
					min_priority = stack.peek().priority
					min_node_pattern = pattern 
			if min_node_pattern is None:
				raise Exception("broken wtf")
			else:
				stack_data = nodes[min_node_pattern].pop()
				for transition in min_node_pattern.transitions[0].transitions:
					if transition.next_pattern in nodes:
						#TODO: change this to not a compound check
						i = len(nodes[transition.next_pattern])
						found = False
						while i > 0 and not found:
							if transition.next_pattern not in nodes[transition.next_pattern].stack[i].parents:
								nodes[transition.next_pattern].stack[i].parents[transition.next_pattern] = Node()
								nodes[transition.next_pattern].stack[i].priority = transition.priority
								found = True
							i -= 1
					else:
						nodes[transition.next_pattern] = Node.Stack()
			iterations += 1
		return input_nodes 

class NewNode():
	def init(self):
		self.index: int = 0
		self.id: int = 0
		self.node_pattern: NodePattern = NodePattern()
		self.children: Set[NewNode] = set()
		self.parents: Set[NewNode] = set()
		self.output_shape: Size = Size()
		self.mould_shape: Size = Size()
	def add_child(self, child: NewNode) -> None:
		if child not in self.children:
			self.children.add(child)
			child.add_parent(self)
	def add_parent(self, parent: NewNode) -> None:
		if parent not in self.parents:
			self.parents.add(parent)
			parent.add_child(self)
	def to_flat_module(self) -> None:
		pass
	def to_runnable_graph(self) -> None:
		pass

