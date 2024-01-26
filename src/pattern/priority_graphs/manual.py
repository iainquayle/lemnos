from __future__ import annotations
from dataclasses import dataclass

from src.pattern.node_parameters import BaseParameters, IdentityParameters 
from src.shared.shape import Bound, LockedShape, Shape, OpenShape
from src.shared.index import Index
from src.shared.merge_method import MergeMethod, Concat
from src.model.graph import Node, Graph as ModelGraph

from abc import abstractmethod
from typing import List, Set, Dict, Tuple, Iterable
from copy import copy

#TODO: items that need to be added:
#	macro parameters, only a certain number of these can be used? maybe in a chain, somehow relate to other nodes

@dataclass
class _ExpansionNode:
	def __init__(self, parents: List[Node], priority: int) -> None:
		self.parents: List[Node] = parents #may be quicker to make this a dict again
		self.priority: int
	def get_parent_shapes(self) -> List[LockedShape]:
		return [parent.get_output_shape() for parent in self.parents]
	def add_parent(self, parent: Node, priority: int) -> None:
		if self.taken(parent):
			raise ValueError("Parent already taken")
		self.parents.append(parent)
		self.priority = min(self.priority, priority) 
	def taken(self, node: Node) -> bool:
		for parent in self.parents:
			if parent.get_pattern() == node.get_pattern():
				return True
		return False
	def __copy__(self) -> _ExpansionNode:
		return _ExpansionNode(copy(self.parents), self.priority)
class _ExpansionStack:
	__slots__ = ["_stack"]
	def __init__(self, stack: List[_ExpansionNode] = []) -> None:
		self._stack: List[_ExpansionNode] = stack 
	def push(self, data: _ExpansionNode) -> None:
		self._stack.append(data)
	def get_available(self, node: Node) -> _ExpansionNode | None: 
		for i in range(len(self._stack) - 1, -1, -1):
			if not self._stack[i].taken(node):
				return self._stack[i] 
		return None
	def pop(self) -> _ExpansionNode:
		return self._stack.pop()
	def peek(self) -> _ExpansionNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().priority
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _ExpansionStack:
		return _ExpansionStack(copy(self._stack))
def _min_expansion(expansion_nodes: Dict[NodePattern, _ExpansionStack]) -> NodePattern | None:
	return min(expansion_nodes.items(), key=lambda item: item[1].get_priority())[0]

class Graph:
	def __init__(self) -> None:
		if len(self._starts) == 0 or len(self._ends) == 0:
			raise ValueError("No start or end patterns")
		self._starts: List[NodePattern] = []
		self._ends: List[NodePattern] = []
	def add_start(self, pattern: NodePattern) -> None:
		self._starts.append(pattern)
	def add_end(self, pattern: NodePattern) -> None:
		self._ends.append(pattern)
	#TODO: perhaps add a flag that switches whether the indices should be attempted in order, or just used at random for breeding,
	#	could use a random seed, that can also effectivley work as a flag
	#only real option for capturing input and output nodes in current setup is to return a list of nodes and find them after
	#TODO: consider making it such that start patterns must have single transitions to unique patterns
	#	or make join_existing a enum, with a third option of auto, which will make or join
	#	since rn inputs dont have priorities, meaning the creator of a node wont necessarily go first
	#	or just try finding the true start pattern
	def build(self, input_shapes: List[LockedShape], output_shapes: List[LockedShape], indices: List[Index]) -> None:
		if len(input_shapes) != len(self._starts):
			raise ValueError("Incorrect number of input shapes")
		expansion_nodes: Dict[NodePattern, _ExpansionStack] = {}
		input_nodes: List[Node] = []
		for i, (pattern, shape) in enumerate(zip(self._starts, input_shapes)):
			input = NodePattern(IdentityParameters(Bound([None] * len(shape))), Concat())
			input_node = Node(Index(), i, input, shape, shape, None)
			input_nodes.append(input_node)
			expansion_nodes[pattern] = _ExpansionStack([_ExpansionNode([input_node], -1)])
		chosen: NodePattern | None = _min_expansion(expansion_nodes) 
		if chosen is None:
			raise ValueError("No valid start pattern")	
		else: 
			chosen.build(expansion_nodes, indices, len(input_nodes))
	def _build_node(self, expansion_nodes: Dict[NodePattern, _ExpansionStack], indices: List[Index], id: int) -> List[Node] | NodePattern:
		chosen: NodePattern | None = _min_expansion(expansion_nodes)
		if chosen is None:
			return []
		else:

			return []	
		

class NodePattern:
	__slots__ = ["_node_parameters", "_transition_groups", "_merge_method"]
	def __init__(self, node_parameters: BaseParameters, merge_method: MergeMethod) -> None:
		self._transition_groups: List[TransitionGroup] = []
		self._node_parameters: BaseParameters = node_parameters 
		self._merge_method: MergeMethod = merge_method 
	def add_transition_group(self, group: TransitionGroup) -> None:
		self._transition_groups.append(copy(group))
	#procedure
	#	iterate over groups
	#		check immediate constraints 
	#			expand node
	#			recurse
	#			if fail comes back, try next group
	#		if all fail, return node pattern 
	#			maybe node itself? would node pattern possibly destroy whole graph?
	#			it shouldnt, as the node that expanded was at the top of the stack
	#			and any node invalidated by it would have been removed on the way back up apart from the original creator
	def build(self, expansion_nodes: Dict[NodePattern, _ExpansionStack], indices: List[Index], id: int) -> List[Node] | NodePattern:
		index = indices[0]
		parents: Iterable[Node] = expansion_nodes[self].pop().parents
		input_shape = self._merge_method.get_output_shape([parent.get_output_shape() for parent in parents])
		pivot = index.to_int(len(self._transition_groups))
		i = 0
		while abs(i) < max(len(self._transition_groups) - pivot, pivot): 
			if pivot + i < len(self._transition_groups) and pivot + i >= 0:
				group = self._transition_groups[pivot + i]
				
			i = -i if i > 0 else -i + 1


		valid_group_and_shapes: Tuple[TransitionGroup, Dict[Transiton, _ExpansionNode], Shape, Shape] | None = None
		group_iter = iter(self._transition_groups)
		while (group := next(group_iter, None)) is not None: #add cut off once reached past index point
			conformance_shape = OpenShape.new()
			join_nodes: Dict[Transiton, _ExpansionNode] = {}
			transition_iter = iter(group._transitions)
			while (transition := next(transition_iter, None)) is not None and conformance_shape is not None:
				if transition.join_existing and ((join_on := expansion_nodes[transition.next].get_available(parents[0])) is not None
						and (conformance_shape := conformance_shape.common_lossless(transition.next.get_conformance_shape(join_on.get_parent_shapes()))) is not None):
					join_nodes[transition] = join_on
				else:
					conformance_shape = None #bit of a hack but technically true
			if conformance_shape is not None:
				shapes = self._node_parameters.get_mould_and_output_shapes(input_shape, conformance_shape, index)
				if shapes is not None:
					valid_group_and_shapes = (group, join_nodes, *shapes)
		if valid_group_and_shapes is None:
			return self 
		else:
			group, join_nodes, mould_shape, output_shape = valid_group_and_shapes
			node = Node(index, id, self, mould_shape, output_shape, parents)
			for transition in valid_group_and_shapes[0]._transitions:
				stack = expansion_nodes[transition.next]
				if transition.join_existing:
					join_nodes[transition].add_parent(node, transition.priority)
				else:
					if transition.next not in expansion_nodes:
						expansion_nodes[transition.next] = _ExpansionStack()
					stack.push(_ExpansionNode([node], transition.priority))
			return [node]
	#def _build_group(self, group_index: int)
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
	def __init__(self, next: NodePattern, priority: int, join_existing: bool = False) -> None:
		if priority > Transiton._MAX_PRIORITY or priority < Transiton._MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self.next: NodePattern = next
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
		pattern_set: Set[NodePattern] = set()
		for transition in transitions:
			if transition.next in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.next)
		self._transitions = transitions
	def __str__(self) -> str:
		return f"TG{self._transitions}"
	def __repr__(self) -> str:
		return str(self)
