from __future__ import annotations
from dataclasses import dataclass

from src.pattern.node_parameters import BaseParameters 
from src.shared.shape import Bound, LockedShape, Shape
from src.shared.index import Index

from abc import abstractmethod
from typing import List, Set, Dict
from copy import copy

#TODO: items that need to be added:
#	macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes

@dataclass
class _ExpansionNode:
	parents: Dict[NodePattern, Node]
	priority: int
	def get_conformance_shape(self) -> Shape | None:
		return Shape.reduce_common_lossless([parent.get_output_shape() for parent in self.parents.values()])
	def __copy__(self) -> _ExpansionNode:
		return _ExpansionNode(copy(self.parents), self.priority)
class _ExpansionStack:
	__slots__ = ["_stack"]
	def __init__(self, stack: List[_ExpansionNode] = []) -> None:
		self._stack: List[_ExpansionNode] = stack 
	def push(self, data: _ExpansionNode) -> None:
		self._stack.append(data)
	def pop(self) -> _ExpansionNode:
		return self._stack.pop()
	def peek(self) -> _ExpansionNode:
		return self._stack[-1]
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _ExpansionStack:
		return _ExpansionStack(copy(self._stack))

class Graph:
	def __init__(self) -> None:
		self._start_patterns: List[NodePattern] = []
		self._end_patterns: List[NodePattern] = []
	def add_start_pattern(self, pattern: NodePattern) -> None:
		self._start_patterns.append(pattern)
	def add_end_pattern(self, pattern: NodePattern) -> None:
		self._end_patterns.append(pattern)
	#TODO: perhaps add a flag that switches whether the indices should be attempted in order, or just used at random for breeding
	def build(self, input_shapes: List[LockedShape], output_shapes: List[LockedShape], indices: List[Index]) -> None:
		expansion_nodes: Dict[NodePattern, _ExpansionStack] = {}
		pass

class NodePattern:
	__slots__ = ["_node_parameters", "_transition_groups"]
	def __init__(self, node_parameters: BaseParameters):
		self._transition_groups: List[TransitionGroup] = []
		self._node_parameters: BaseParameters = node_parameters 
	def add_transition_group(self, group: TransitionGroup) -> None:
		self._transition_groups.append(copy(group))
	@abstractmethod	
	def analyze(self) -> None:
		#paceholder for possible auto priority assignment
		pass

MAX_PRIORITY: int = 128 
@dataclass
class Transiton:
	next_pattern: NodePattern
	optional: bool = False
	priority: int = 0
	join_existing: bool = False

class TransitionGroup:
	__slots__ = ["_transitions", "_repetition_bounds"]
	def __init__(self, transitions: List[Transiton], repetition_bounds: Bound =Bound())  -> None:
		self._transitions: List[Transiton] = copy(transitions)
		self._repetition_bounds: Bound = repetition_bounds
	def set_transitions(self, transitions: List[Transiton]) -> None:
		pattern_set: Set[NodePattern] = set()
		for transition in transitions:
			if transition.next_pattern in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.next_pattern)
		self._transitions = transitions
	def __str__(self) -> str:
		return f"TG{self._transitions}"
	def __repr__(self) -> str:
		return str(self)
