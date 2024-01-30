from __future__ import annotations

from src.schema.node_parameters import BaseParameters  
from src.shared.shape import Bound, LockedShape, Shape 
from src.shared.merge_method import MergeMethod 

from typing import List, Set 
from copy import copy 

#TODO: items that need to be added:
#	macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes
class Schema:
	def __init__(self) -> None:
		if len(self.inputs) == 0 or len(self.outputs) == 0:
			raise ValueError("No start or end patterns")
		self.inputs: List[SchemaNode] = []
		self.outputs: List[SchemaNode] = []
	def add_start(self, pattern: SchemaNode) -> None:
		self.inputs.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self.outputs.append(pattern)
	#TODO: perhaps add a flag that switches whether the indices should be atModelNodeted in order, or just used at random for breeding,
	#	could use a random seed, that can also effectivley work as a flag
	#	only real option for capturing input and output nodes in current setup is to return a list of nodes and find them after
	#TODO: consider adding ordering to input nodes to it 
	#TODO: consider making turning join existing into enum

class SchemaNode:
	__slots__ = ["node_parameters", "transition_groups", "merge_method"]
	def __init__(self, node_parameters: BaseParameters, merge_method: MergeMethod) -> None:
		self.transition_groups: List[TransitionGroup] = []
		self.node_parameters: BaseParameters = node_parameters 
		self.merge_method: MergeMethod = merge_method 
	def add_transition_group(self, group: TransitionGroup) -> None:
		self.transition_groups.append(copy(group))
	def get_conformance_shape(self, input_shapes: List[LockedShape]) -> Shape:
		return self.merge_method.get_conformance_shape(input_shapes)

class Transition:
	_MAX_PRIORITY: int = 128 
	_MIN_PRIORITY: int = 0 
	__slots__ = ["next", "optional", "priority", "join_existing"]
	def __init__(self, next: SchemaNode, priority: int, join_existing: bool = False) -> None:
		if priority > Transition._MAX_PRIORITY or priority < Transition._MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self.next: SchemaNode = next
		self.optional: bool =  False 
		self.priority: int = priority 
		self.join_existing: bool = join_existing 
	@staticmethod
	def get_max_priority() -> int:
		return Transition._MAX_PRIORITY 
	@staticmethod
	def get_min_priority() -> int:
		return Transition._MIN_PRIORITY

class TransitionGroup:
	__slots__ = ["transitions", "repetition_bounds"]
	def __init__(self, transitions: List[Transition], repetition_bounds: Bound = Bound()) -> None:
		self.transitions: List[Transition] = copy(transitions)
		self.repetition_bounds: Bound = repetition_bounds
	def set_transitions(self, transitions: List[Transition]) -> None:
		pattern_set: Set[SchemaNode] = set()
		for transition in transitions:
			if transition.next in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.next)
		self.transitions = transitions
	def __str__(self) -> str:
		return f"TG{self.transitions}"
	def __repr__(self) -> str:
		return str(self)
