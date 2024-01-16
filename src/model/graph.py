from __future__ import annotations

import torch
from torch import Size 

from src.pattern.priority_graphs.manual import NodePattern, MAX_PRIORITY, Graph as BuildGraph 
from src.pattern.commons import Index
from src.shared.shape import Shape 

from typing import Dict, List, Set, Tuple, Iterable
from typing_extensions import Self
from dataclasses import dataclass

class Graph():
	def __init__(self, input_nodes: List[Node] = list(), output_nodes: List[Node] = list()) -> None:
		self._input_nodes: List[Node] = input_nodes 
		self._output_nodes: List[Node] = output_nodes 
	def to_flat_source_module(self) -> Tuple[str, str]:
		return "", ""
	def to_runnable_module(self) -> None:
		pass
	@dataclass
	class ExpansionNode:
		parents: Dict[NodePattern, Node]
		priority: int
		def get_conformance_shape(self) -> Shape:
			#TODO: implement
			return Shape.fixed(1)
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
	def init(self, index: Index, id: int, node_pattern: NodePattern, output_shape: Shape, mould_shape: Shape, parents: Iterable[Self] | None) -> None:
		self._index: Index = index
		self._id: int = id 
		self._node_pattern: NodePattern = node_pattern 
		self._children: Set[Self] = set()
		self._parents: Set[Self] = set()
		if parents is not None:
			self.set_parents(parents)
		self._output_shape: Shape = output_shape
		self._mould_shape: Shape = mould_shape 
	def add_child(self, child: Self) -> None:
		if child not in self._children:
			self._children.add(child)
			child.add_parent(self)
	def add_parent(self, parent: Self) -> None:
		if parent not in self._parents:
			self._parents.add(parent)
			parent.add_child(self)
	def set_parents(self, parents: Iterable[Self]) -> None:
		self._parents = set()
		for parent in parents:
			self.add_parent(parent)
