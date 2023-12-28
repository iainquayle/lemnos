from __future__ import annotations

import torch
from torch import Tensor, Size 
from torch.nn import Module, ModuleList
import torch.nn as nn

from src.model_structures.commons import  get_features_shape, mould_features
from src.build_structures.priority_graphs.manual import NodePattern, MAX_PRIORITY, Graph as BuildGraph 

from typing import Dict, List, Set, Tuple
from typing_extensions import Self
from dataclasses import dataclass

#TODO: need to implement some means by which a node can check what dims a node may already have that it is connecting to

class Graph():
	def __init__(self, input_nodes: List[Node] = list(), output_nodes: List[Node] = list()) -> None:
		self.input_nodes: List[Node] = input_nodes 
		self.output_nodes: List[Node] = output_nodes 
	def to_flat_source_module(self) -> Tuple[str, str]:
		return "", ""
	def to_runnable_module(self) -> None:
		pass
	@dataclass
	class ExpansionNode:
		parents: Dict[NodePattern, Node]
		priority: int
	class ExpansionStack:
		def __init__(self) -> None:
			self.stack: List[Graph.ExpansionNode] = [Graph.ExpansionNode(dict(), 0)]
		def push(self, data: Graph.ExpansionNode) -> None:
			self.stack.append(data)
		def pop(self) -> Graph.ExpansionNode:
			return self.stack.pop()
		def peek(self) -> Graph.ExpansionNode:
			return self.stack[-1]
		def __len__(self) -> int:
			return len(self.stack)
	#TODO: consider there being no explicit shapes in and out, instead have it entirely dealt with by the node patterns
	#TODO: a recursive build instead, to allow for backtracking, rather than trying to force shapes
	#	only other option is some type of lookahead, which would be a pain
	@staticmethod
	def from_build_graph(build_graph: BuildGraph, shapes_in: List[Size], shape_outs: List[Size]) -> Graph:
		input_nodes: List[Node] = []
		output_nodes: List[Node] = []
		expansion_nodes: Dict[NodePattern, Graph.ExpansionStack] = {}
		for pattern in build_graph.start_patterns:
			node = Node()
			input_nodes.append(node)
			expansion_nodes[pattern] = Graph.ExpansionStack()
		MAX_ITERATIONS = 1024 
		iterations = 0
		while len(expansion_nodes) > 0 and iterations < MAX_ITERATIONS:
			min_node_pattern = None 
			min_priority = MAX_PRIORITY + 1
			for pattern, stack in expansion_nodes.items():
				if stack.peek().priority < min_priority:
					min_priority = stack.peek().priority
					min_node_pattern = pattern 
			if min_node_pattern is None:
				raise Exception("broken wtf")
			else:
				#min chosen, and which to expand
				#remove 
				stack_data = expansion_nodes[min_node_pattern].pop()
				if len(expansion_nodes[min_node_pattern]) == 0:
					del expansion_nodes[min_node_pattern]
				expanded = False 
				transition_group_iter = iter(min_node_pattern.transition_groups)
				while (transition_group := next(transition_group_iter, None)) is not None and not expanded:
					#need to be able to check that all transitions are valid, and if not then skip
					#to do so, maybe make get_output_shape return None if it is not valid
					output_shapes = [] 
					for transition in transition_group.transitions:
						if transition.next_pattern in expansion_nodes:
							i = len(expansion_nodes[transition.next_pattern])
							while i > 0 and transition.next_pattern not in expansion_nodes[transition.next_pattern].stack[i].parents:
								i -= 1
							if i >= 0:
								expansion_nodes[transition.next_pattern].stack[i].parents[transition.next_pattern] = Node()
								expansion_nodes[transition.next_pattern].stack[i].priority = transition.priority
							else:
								pass
						else:
							expansion_nodes[transition.next_pattern] = Graph.ExpansionStack()
				#if
			iterations += 1
		graph = Graph(input_nodes, [])
		return graph 

class Node():
	def init(self):
		self.index: int = 0
		self.id: int = 0
		self.node_pattern: NodePattern = NodePattern()
		self.children: Set[Node] = set()
		self.parents: Set[Node] = set()
		self.output_shape: Size = Size()
		self.mould_shape: Size = Size()
	def add_child(self, child: Node) -> None:
		if child not in self.children:
			self.children.add(child)
			child.add_parent(self)
	def add_parent(self, parent: Node) -> None:
		if parent not in self.parents:
			self.parents.add(parent)
			parent.add_child(self)
