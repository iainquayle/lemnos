from __future__ import annotations

from src.model.model import ModelNode, Model
from src.shared.shape import LockedShape, OpenShape, Shape
from src.shared.index import Index
from src.schema.schema_node import SchemaNode, Transition, TransitionGroup

from typing import List, Dict, Tuple, Iterable

from copy import copy

#TODO: move builder into schema, and seperate model and model nodes into their own files
#	this will make the schema and model more parallel in structure

#TODO: consider making turning join existing into enum
class OtherBuildIndices:
	#while this may seem the best, how would mutation work?
	#	perhaps have hyperparam that dictates the likelihood of mutation
	#	much like the likelihood of a jump to a different sequence
	__slots__ = ["_sequences"]
	def __init__(self, sequences: List[List[Tuple[Index, SchemaNode, Shape]]] = []) -> None:
		#index, schema, shape in
		self._sequences: List[List[Tuple[Index, SchemaNode, Shape]]] = sequences
	def get_index(self, sequence: int, id: int, schema_node: SchemaNode, shape: Shape) -> Index:
		if sequence < len(self._sequences) and id < len(self._sequences[sequence]):
			pass
		return Index()


class BuildIndices:
	__slots__ = ["_indices", "_pool"]
	def __init__(self, sequences: List[List[Tuple[Index, SchemaNode]]] = [], pool: List[Tuple[Index, SchemaNode]] = []) -> None:
		self._indices: List[List[Tuple[Index, SchemaNode]]] = sequences 
		self._pool: List[Tuple[Index, SchemaNode]] = pool
	def get_index(self, sequence: int, id: int, schema_node: SchemaNode) -> Index:
		if sequence < len(self._indices) and id < len(self._indices[sequence]): 
			#TODO: add some flexibilty, allow it to search within a range for a schema node that matches
			return self._indices[sequence][id][0]
		else:
			return Index()

class ModelBuilder:
	def __init__(self, inputs: List[SchemaNode], outputs: List[SchemaNode], max_nodes: int = 1024) -> None:
		if len(inputs) == 0 or len(outputs) == 0:
			raise ValueError("No start or end patterns")
		self.inputs: List[SchemaNode] = inputs 
		self.outputs: List[SchemaNode] = outputs 
		self.max_nodes: int = max_nodes
	def add_start(self, pattern: SchemaNode) -> None:
		self.inputs.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self.outputs.append(pattern)
	#options
	#	always have a pool of indices available when not building entirley from a known list
	#		always should have schema nodes attached
	#		if starting fresh then guess they need to be optional then
	#	pass singular list of indices, and a pool to choose from when the list is exhausted
	#		this case would not need the schema nodes attached
	#		unless it wanted to be allowed to split the list and use the schema nodes to line them back up
	#		decent option, would technically work
	#	pass in multiple lists, mix and match sequences as best as possible
	#		need to have schema nodes attached, try and line them back up 
	#		likely the best option for breeding, gives the most flexibility
	#		the pool could be not matched with schema nodes, and be the functionality of mutation and gap filling
	#		List[List[Tuple[Index, SchemaNode]]] 
	#	pass only a pool, paired with model nodes or atleast a size
	#		if the size is somewhat the same, then it may be be a better means of carrying across traits from one model to another
	#could implement a number of these?
	def build(self, input_shapes: List[LockedShape], indices: BuildIndices) -> Model | None:
		if len(input_shapes) != len(self.inputs):
			raise ValueError("Incorrect number of input shapes")
		nodes = _BuildTracker.build_nodes({input_schema: shape for input_schema, shape in zip(self.inputs, input_shapes)}, indices, self.max_nodes)
		if nodes is not None: #TODO: this needs to be checked
			input_nodes = [node for node in nodes if node.get_schema_node() in self.inputs and len(node.get_parents()) == 0]
			output_nodes = [node for node in nodes if node.get_schema_node() in self.outputs and len(node.get_children()) == 0]
			if len(input_nodes) == len(self.inputs) and len(output_nodes) == len(self.outputs):
				return Model(input_nodes, output_nodes)
		return None

class _BuildTracker:
	_MAX_NODES = 512 
	__slots__ = ["_stacks", "_max_nodes", "_indices", "_node_counts", "_sequence", "_sequence_offset"]
	def __init__(self, indices: BuildIndices, max_nodes: int, stacks: Dict[SchemaNode, _BuildStack] = dict()) -> None:
		self._stacks: Dict[SchemaNode, _BuildStack] = stacks 
		self._node_counts: Dict[SchemaNode, int] = {}
		self._max_nodes: int = max_nodes
		self._indices: BuildIndices = indices
		self._sequence: int = 0
		self._sequence_offset: int = 0
		#perhaps change the index schema node pairing to use a model node, to keep track of the size
		#move depth to tracking here
		#track sequence used currently
		#perhaps also track the offset of the depth required to find an index that fits
	@staticmethod
	def build_nodes(inputs: Dict[SchemaNode, LockedShape], indices: BuildIndices, max_nodes: int) -> List[ModelNode] | None:
		dummy_nodes = {input_schema: ModelNode(Index(), -1, input_schema, shape, shape, None) for input_schema, shape in inputs.items()}
		tracker = _BuildTracker(indices, max_nodes, {input_schema: _BuildStack([_BuildNode([dummy_node], -1)]) for input_schema, dummy_node in dummy_nodes.items()})
		if isinstance((result := tracker._build_min(indices, 0)), List):
			for node in dummy_nodes.values():
				node.unbind()
			return result
		return None
	def _build_min(self, indices: BuildIndices, depth: int) -> List[ModelNode] | SchemaNode:
		if (result := self._pop_min_node()) is not None:
			schema_node, build_node = result
			index = Index()
			parents = build_node.get_parents()
			mould_shape = schema_node.get_mould_shape([parent.get_output_shape() for parent in parents])
			pivot = index.get_shuffled(len(schema_node.get_transition_groups()))
			i = 0
			while abs(i) <= max(len(schema_node.get_transition_groups()) - pivot, pivot):
				if pivot + i < len(schema_node.get_transition_groups()) and pivot + i >= 0:
					group = schema_node[pivot + i]
					conformance_shape = self._get_group_conformance_shape(group, schema_node)
					if conformance_shape is not None:
						tracker_copy = copy(self)
						if (output_shape := schema_node.get_output_shape(mould_shape, conformance_shape, index)) is not None:
							node = ModelNode(index, depth, schema_node, mould_shape, output_shape, parents)
							self._increment_count(schema_node)
							if (depth < self._max_nodes 
			   						and tracker_copy._record_transitions(iter(group), node) 
			   						and isinstance(result := tracker_copy._build_min(indices, depth + 1), List)):
								return [node, *result]
							else:
								node.unbind()	
				i = -i if i > 0 else -i + 1
			if len(schema_node.get_transition_groups()) == 0:
				if (output_shape := schema_node.get_output_shape(mould_shape, OpenShape(), index)) is not None:
					return [ModelNode(index, depth, schema_node, mould_shape, output_shape, parents)]
			return schema_node
		return []
	def _increment_count(self, schema_node: SchemaNode) -> None:
		self._node_counts[schema_node] = self._node_counts.get(schema_node, 0) + 1
	def _get_count(self, schema_node: SchemaNode) -> int:
		return self._node_counts.get(schema_node, 0)
	def _get_group_conformance_shape(self, group: TransitionGroup, schema_node: SchemaNode) -> Shape | None:
		transition_iter = iter(group)
		conformance_shape = OpenShape()
		while (transition := next(transition_iter, None)) is not None and conformance_shape is not None: #TODO: simplify somehow, fugly
			if transition.get_join_existing():
				if (join_node := self[transition.get_next()].get_available(schema_node)) is not None: 
					conformance_shape = conformance_shape.common_lossless(transition.get_next().get_conformance_shape(join_node.get_parent_shapes()))
				else:
					conformance_shape = None
		return conformance_shape
	def _min_stack(self) -> Tuple[SchemaNode, _BuildStack] | None: 
		if len(self) == 0:
			return None
		min_schema = min(self.get_iter(), key=lambda item: item[1].get_priority()) 
		if len(min_schema[1]) == 0:
			return None
		return min_schema
	def _pop_min_node(self) -> Tuple[SchemaNode, _BuildNode] | None:
		if (result := self._min_stack()) is not None:
			schema, stack = result
			return schema, stack.pop()
		return None
	def _record_transitions(self, transitions: Iterable[Transition], parent: ModelNode) -> bool:
		for transition in transitions:
			if not self.record_transition(transition, parent):
				return False
		return True
	def record_transition(self, transition: Transition, parent: ModelNode) -> bool:
		if transition.get_join_existing():
			if transition.get_next() in self and (join_on_node := self[transition.get_next()].get_available(parent)) is not None:
				join_on_node.add_parent(parent, transition.get_priority())
				return True
			else:
				return False
		else:
			if transition.get_next() not in self:
				self[transition.get_next()] = _BuildStack([_BuildNode([parent], transition.get_priority())])
			else:
				self[transition.get_next()].push(_BuildNode([parent], transition.get_priority()))
			return True	
	def is_empty(self) -> bool:
		for _, stack in self.get_iter():
			if len(stack) > 0:
				return False
		return True
	def stacks_str(self) -> str:
		return " , ".join([schema.debug_name + ": " + str(len(stack)) for schema, stack in self.get_iter()])
	def __getitem__(self, key: SchemaNode) -> _BuildStack:
		return self._stacks[key]
	def __setitem__(self, key: SchemaNode, value: _BuildStack) -> None:
		self._stacks[key] = value
		return
	def __copy__(self) -> _BuildTracker:
		return _BuildTracker(self._indices, self._max_nodes, {key: copy(value) for key, value in self.get_iter()})
	def next_tracker(self) -> _BuildTracker:
		return _BuildTracker(self._indices, self._max_nodes, {key: copy(value) for key, value in self.get_iter()})
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._stacks
	def __len__(self) -> int:
		return len(self._stacks)
	def get_iter(self) -> Iterable[Tuple[SchemaNode, _BuildStack]]:
		return iter(self._stacks.items())
class _BuildNode:
	__slots__ = ["_parents", "_priority"]
	def __init__(self, parents: List[ModelNode], priority: int) -> None:
		self._parents: Dict[SchemaNode, ModelNode] = {parent.get_schema_node(): parent for parent in parents} 
		self._priority: int = priority 
	def get_parent_shapes(self) -> List[LockedShape]:
		return [parent.get_output_shape() for parent in self._parents.values()]
	def get_parents(self) -> List[ModelNode]:
		return list(self._parents.values())
	def get_priority(self) -> int:
		return self._priority
	def add_parent(self, parent: ModelNode, priority: int) -> bool: 
		if not self.available(parent):
			return False
		self._parents[parent.get_schema_node()] = parent
		self._priority = min(self._priority, priority) 
		return True
	def available(self, parent: ModelNode | SchemaNode) -> bool:
		return (parent.get_schema_node() if isinstance(parent, ModelNode) else parent) not in self._parents 
	def __copy__(self) -> _BuildNode:
		return _BuildNode(copy(self.get_parents()), self._priority)
class _BuildStack:
	__slots__ = ["_stack"]
	def __init__(self, stack: List[_BuildNode] = []) -> None:
		self._stack: List[_BuildNode] = stack 
	def push(self, data: _BuildNode) -> None:
		self._stack.append(data)
	def get_available(self, parent: ModelNode | SchemaNode) -> _BuildNode | None: 
		result = None
		for node in self._stack:
			if node.available(parent):
				result = node
		return result
	def pop(self) -> _BuildNode:
		return self._stack.pop()
	def peek(self) -> _BuildNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().get_priority() if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _BuildStack:
		return _BuildStack([copy(node) for node in self._stack])
