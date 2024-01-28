from __future__ import annotations
from dataclasses import dataclass

from src.schema.node_parameters import BaseParameters, IdentityParameters 
from src.shared.shape import Bound, LockedShape, Shape, OpenShape
from src.shared.index import Index
from src.shared.merge_method import MergeMethod, Concat
from src.model.model import ModelNode, Model

from abc import abstractmethod
from typing import List, Set, Dict, Tuple, Iterable
from copy import copy, deepcopy 

#TODO: items that need to be added:
#	macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes

@dataclass
class _ExpansionModelNode:
	def __init__(self, parents: List[ModelNode], priority: int) -> None:
		self.parents: List[ModelNode] = parents #may be quicker to make this a dict again
		self.priority: int = priority 
	def get_parent_shapes(self) -> List[LockedShape]:
		return [parent.get_output_shape() for parent in self.parents]
	def add_parent(self, parent: ModelNode, priority: int) -> None:
		if self.taken(parent):
			raise ValueError("Parent already taken")
		self.parents.append(parent)
		self.priority = min(self.priority, priority) 
	def taken(self, node: ModelNode) -> bool:
		for parent in self.parents:
			if parent.get_pattern() == node.get_pattern():
				return True
		return False
	def __copy__(self) -> _ExpansionModelNode:
		return _ExpansionModelNode(copy(self.parents), self.priority)
class _ExpansionStack:
	__slots__ = ["_stack"]
	def __init__(self, stack: List[_ExpansionModelNode] = []) -> None:
		self._stack: List[_ExpansionModelNode] = stack 
	def push(self, data: _ExpansionModelNode) -> None:
		self._stack.append(data)
	def get_available(self, node: ModelNode) -> _ExpansionModelNode | None: 
		for i in range(len(self._stack) - 1, -1, -1):
			if not self._stack[i].taken(node):
				return self._stack[i] 
		return None
	def pop(self) -> _ExpansionModelNode:
		return self._stack.pop()
	def peek(self) -> _ExpansionModelNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().priority
	def __len__(self) -> int:
		return len(self._stack)
	def __deepcopy__(self) -> _ExpansionStack:
		return _ExpansionStack(copy(self._stack))
class _ExpansionCollection:
	__slots__ = ["_expansion_nodes"]
	def __init__(self, expansion_nodes: Dict[SchemaNode, _ExpansionStack] = dict()) -> None:
		self._expansion_nodes: Dict[SchemaNode, _ExpansionStack] = expansion_nodes
	def min(self) -> Tuple[SchemaNode, _ExpansionStack] | None:
		return min(self._expansion_nodes.items(), key=lambda item: item[1].get_priority())
	def add(self, node: ModelNode, priority: int) -> None:
		if node.get_pattern() in self._expansion_nodes:
			self._expansion_nodes[node.get_pattern()].push(_ExpansionModelNode([node], priority))
		else:
			self._expansion_nodes[node.get_pattern()] = _ExpansionStack([_ExpansionModelNode([node], priority)])
	def __getitem__(self, key: SchemaNode) -> _ExpansionStack:
		return self._expansion_nodes[key]
	def __copy__(self) -> _ExpansionCollection:
		return _ExpansionCollection(deepcopy(self._expansion_nodes))
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._expansion_nodes
	
#TODO: rename to schema
class Schema:
	def __init__(self) -> None:
		if len(self._starts) == 0 or len(self._ends) == 0:
			raise ValueError("No start or end patterns")
		self._starts: List[SchemaNode] = []
		self._ends: List[SchemaNode] = []
	def add_start(self, pattern: SchemaNode) -> None:
		self._starts.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self._ends.append(pattern)
	#TODO: perhaps add a flag that switches whether the indices should be atModelNodeted in order, or just used at random for breeding,
	#	could use a random seed, that can also effectivley work as a flag
	#	only real option for capturing input and output nodes in current setup is to return a list of nodes and find them after
	#TODO: consider adding ordering to input nodes to it 
	#TODO: consider making turning join existing into enum
	def build(self, input_shapes: List[LockedShape], output_shapes: List[LockedShape], indices: List[Index]) -> Model:
		if len(input_shapes) != len(self._starts):
			raise ValueError("Incorrect number of input shapes")
		expansion_collection: _ExpansionCollection = _ExpansionCollection()
		input_nodes: List[ModelNode] = []
		for i, shape in enumerate(input_shapes):
			input = SchemaNode(IdentityParameters(Bound([None] * len(shape))), Concat())
			input_node = ModelNode(Index(), i, input, shape, shape, None)
			input_nodes.append(input_node)
			expansion_collection.add(input_node, -1)
		if (result := expansion_collection.min()) is not None:
			to_expand, _ = result
			to_expand.build(expansion_collection, indices, len(input_nodes))
		else:
			raise ValueError("No valid input")

class SchemaNode:
	__slots__ = ["_node_parameters", "_transition_groups", "_merge_method"]
	def __init__(self, node_parameters: BaseParameters, merge_method: MergeMethod) -> None:
		self._transition_groups: List[TransitionGroup] = []
		self._node_parameters: BaseParameters = node_parameters 
		self._merge_method: MergeMethod = merge_method 
	def add_transition_group(self, group: TransitionGroup) -> None:
		self._transition_groups.append(copy(group))
	def build(self, expansion_collection: _ExpansionCollection, indices: List[Index], id: int) -> List[ModelNode] | SchemaNode:
		index = indices[0]
		parents: Iterable[ModelNode] = expansion_collection[self].pop().parents
		input_shape = self._merge_method.get_output_shape([parent.get_output_shape() for parent in parents])
		pivot = index.to_int(len(self._transition_groups))
		i = 0
		while abs(i) < max(len(self._transition_groups) - pivot, pivot): 
			if pivot + i < len(self._transition_groups) and pivot + i >= 0:
				group = self._transition_groups[pivot + i]
				transition_iter = iter(group._transitions)
				join_nodes: Dict[Transiton, _ExpansionModelNode] = {}
				conformance_shape = OpenShape.new()
				while (transition := next(transition_iter, None)) is not None and conformance_shape is not None:
					if transition.join_existing:
						if join_on := expansion_collection[transition.next].get_available(parents[0]):
							conformance_shape = conformance_shape.common_lossless(transition.next.get_conformance_shape(join_on.get_parent_shapes()))
							join_nodes[transition] = join_on
						else:
							conformance_shape = None
				if conformance_shape is not None:
					new_collection = copy(expansion_collection) 
					shapes = self._node_parameters.get_mould_and_output_shapes(input_shape, conformance_shape, index)
					if shapes is not None:
						node = ModelNode(index, id, self, *shapes, parents)
						for transition in group._transitions:
							stack = new_collection[transition.next]
							if transition.join_existing:
								join_nodes[transition].add_parent(node, transition.priority)
							else:
								new_collection.add(node, transition.priority)
						if (result := new_collection.min()) is not None:
							to_expand, _ = result
							result = to_expand.build(new_collection, indices, id + 1)
							if isinstance(result, SchemaNode):
								for transition in group._transitions:
									if transition.next == result and not transition.join_existing:
										return result
							else:
								return [node, *result]
			i = -i if i > 0 else -i + 1
		return self
	def get_conformance_shape(self, input_shapes: List[LockedShape]) -> Shape:
		return self._merge_method.get_conformance_shape(input_shapes)
	@abstractmethod	
	def analyze(self) -> None:
		#paceholder for possible auto priority assignment
		pass

class Transiton:
	_MAX_PRIORITY: int = 128 
	_MIN_PRIORITY: int = 0 
	__slots__ = ["next", "optional", "priority", "join_existing"]
	def __init__(self, next: SchemaNode, priority: int, join_existing: bool = False) -> None:
		if priority > Transiton._MAX_PRIORITY or priority < Transiton._MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self.next: SchemaNode = next
		self.optional: bool =  False #optional 
		self.priority: int = priority 
		self.join_existing: bool = join_existing 
	@staticmethod
	def get_max_priority() -> int:
		return Transiton._MAX_PRIORITY 
	@staticmethod
	def get_min_priority() -> int:
		return Transiton._MIN_PRIORITY

class TransitionGroup:
	__slots__ = ["_transitions", "_repetition_bounds"]
	def __init__(self, transitions: List[Transiton], repetition_bounds: Bound =Bound())  -> None:
		self._transitions: List[Transiton] = copy(transitions)
		self._repetition_bounds: Bound = repetition_bounds
	def set_transitions(self, transitions: List[Transiton]) -> None:
		pattern_set: Set[SchemaNode] = set()
		for transition in transitions:
			if transition.next in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.next)
		self._transitions = transitions
	def __str__(self) -> str:
		return f"TG{self._transitions}"
	def __repr__(self) -> str:
		return str(self)
