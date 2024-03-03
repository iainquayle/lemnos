from __future__ import annotations

from .schema_node import SchemaNode

from src.shared import Shape, LockedShape, OpenShape, Index
from src.model.model import Model
from src.model.model_node import ModelNode
from .schema_node import SchemaNode, Transition, TransitionGroup

import random
from typing import List, Dict, Tuple, Iterable
from copy import copy
from abc import ABC as Abstract, abstractmethod


class BuildIndices(Abstract):
	@abstractmethod
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> Tuple[Index, int]:	
		pass

class StaticIndices(BuildIndices):
	__slots__ = ["_indices"]
	def __init__(self, indices: List[Index]) -> None:
		self._indices: List[Index] = indices
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> Tuple[Index, int]:
		return self._indices[id], 0 

class BreedIndices(BuildIndices):
	__slots__ = ["_sequences", "_sequence_change_prod", "_mutate_prod"]
	def __init__(self, sequence_change_prod: float = 0, mutate_prod: float = 0, sequences: List[List[Tuple[Index, SchemaNode, LockedShape]]] = []) -> None:
		if sequence_change_prod < 0 or sequence_change_prod > 1 or mutate_prod < 0 or mutate_prod > 1:
			raise ValueError("Invalid probabilities")
		self._sequences: List[List[Tuple[Index, SchemaNode, LockedShape]]] = sequences
		self._sequence_change_prod: float = sequence_change_prod
		self._mutate_prod: float = mutate_prod
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> Tuple[Index, int]:
		def search_sequence(sequence_index: int) -> Tuple[Index, int] | None:
			sequence_index %= len(self._sequences)
			min_diff: int = 2**32
			result: Index | None = None
			for index, node, shape in self._sequences[sequence_index]:
				if node == schema_node and (diff := shape.get_upper_diff(shape_in)) < min_diff:
					min_diff = diff 
					result = index 
			if result is not None:
				return result, min_diff 
			else:
				return None
		if random.random() > self._mutate_prod and len(self._sequences) != 0:
			if random.random() > self._sequence_change_prod:
				if (result := search_sequence(sequence_index)) is not None:
					index, _ = result
					return index, sequence_index
			sequence_indices: List[int] = list(range(len(self._sequences)))
			random.shuffle(sequence_indices)
			for sequence in sequence_indices:
				if (result := search_sequence(sequence)) is not None:
					index, _ = result
					return index, sequence 
		return Index.random(), sequence_index 

#TODO: consider turning join existing into enum
class Schema:
	def __init__(self, inputs: List[SchemaNode], outputs: List[SchemaNode], max_nodes: int = 1024) -> None:
		if len(inputs) == 0 or len(outputs) == 0:
			raise ValueError("No start or end patterns")
		for output in outputs:
			if len(output.get_transition_groups()) > 0:
				raise ValueError("End patterns should not have transition groups")
		self.inputs: List[SchemaNode] = inputs 
		self.outputs: List[SchemaNode] = outputs 
		self.max_nodes: int = max_nodes
	def add_start(self, pattern: SchemaNode) -> None:
		self.inputs.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self.outputs.append(pattern)
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
	__slots__ = ["_stacks", "_max_nodes", "_indices", "_node_counts", "_sequence_index"]
	def __init__(self, indices: BuildIndices, max_nodes: int, stacks: Dict[SchemaNode, _BuildStack] = dict()) -> None:
		self._stacks: Dict[SchemaNode, _BuildStack] = stacks 
		self._node_counts: Dict[SchemaNode, int] = {}
		self._max_nodes: int = max_nodes
		self._indices: BuildIndices = indices
		self._sequence_index: int = 0
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
			parents = build_node.get_parents()
			mould_shape = schema_node.get_mould_shape([parent.get_output_shape() for parent in parents])
			#index = Index()
			#TODO: this one here chief, gotta get the proper index
			index, self._sequence_index = indices.get_index(depth, self._sequence_index, schema_node, mould_shape)
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
