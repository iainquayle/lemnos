from __future__ import annotations

from ..shared import LockedShape, OpenShape, Shape, Index 
from .schema_graph import SchemaNode, TransitionGroup, Transition, JoinType

from typing import List, Tuple, Iterable, Set, Dict

import random
from copy import copy

from abc import ABC as Abstract, abstractmethod
from dataclasses import dataclass

ID = int
@dataclass
class ModuleIR:
	schema_node: SchemaNode
	parent_ids: List[ID]
	transition_id: ID 
	input_shape: LockedShape
	output_shape: LockedShape
	index: Index

class Schema:
	def __init__(self, starts: List[SchemaNode], ends: List[SchemaNode], max_nodes: int = 1024) -> None:
		if len(starts) == 0 or len(ends) == 0:
			raise ValueError("No start or end patterns")
		for end in ends:
			if len(end.get_transition_groups()) > 0:
				raise ValueError("End patterns cannot not have transitions out")
		self._starts: List[SchemaNode] = starts 
		self._ends: List[SchemaNode] = ends 
		self._max_nodes: int = max_nodes
	def add_start(self, pattern: SchemaNode) -> None:
		self._starts.append(pattern)
	def add_end(self, pattern: SchemaNode) -> None:
		self._ends.append(pattern)
	def get_starts_iter(self) -> Iterable[SchemaNode]:
		return iter(self._starts)
	def get_ends_iter(self) -> Iterable[SchemaNode]:
		return iter(self._ends)
	def get_node_with_priority(self) -> List[Tuple[SchemaNode, int]]:
		return [(node, i - len(self._starts)) for i, node in enumerate(self._starts)]
	def compile_IR(self, input_shapes: List[LockedShape], build_indices: NodeCompileIndices, max_nodes: int) -> Tuple[str, StaticIndices]:
		raise NotImplementedError("Direct compile not implemented")


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
				if node == schema_node and (diff := shape.upper_difference(shape_in)) < min_diff:
					min_diff = diff 
					result = index 
			if result is not None:
				return result, min_diff 
			else:
				return None
		if random.random() > self._mutate_prod and len(self._sequences) != 0:
			if random.random() > self._sequence_change_prod or len(self._sequences) == 1:
				if (result := search_sequence(sequence_index)) is not None:
					index, _ = result
					return index, sequence_index
			if len(self._sequences) > 1:
				sequence_indices: List[int] = list(range(sequence_index)) + list(range(sequence_index + 1, len(self._sequences)))
				random.shuffle(sequence_indices)
				for sequence in sequence_indices:
					if (result := search_sequence(sequence)) is not None:
						index, _ = result
						return index, sequence 
		return Index.random(), sequence_index 


class _CompileTracker:
	__slots__ = ["_stacks"]
	def __init__(self, stacks: List[_NodeCompileStack]) -> None:
		self._stacks: List[_NodeCompileStack] = stacks 
	def pop_min(self) -> _CompileNode | None:
		min_stack_index: int = min(range(len(self._stacks)), key=lambda i: self._stacks[i].get_priority())
		if len(self._stacks[min_stack_index]) > 0:
			return self._stacks[min_stack_index].pop()
		return None
	def record_and_get(self, transition_group: TransitionGroup, parent: ModelNode) -> List[ModelNode] | None:
		nodes: List[ModelNode] = []
		for transition in iter(transition_group):
			if transition.get_next() not in self:
				self._stacks[transition.get_next()] = _NodeCompileStack(transition.get_next(), [])
			self._node_counts[transition.get_next()] = self._node_counts.get(transition.get_next(), 0) + 1 
			if (node := self._stacks[transition.get_next()].record_and_get(parent, transition.get_join_type(), transition.get_priority())) is not None:
				nodes.append(node)
			else:
				return None
		return nodes
	def is_empty(self) -> bool:
		for stack in self._stacks:
			if len(stack) > 0:
				return False
		return True
	def stacks_str(self) -> str:
		return " , ".join([schema.debug_name + ": " + str(len(stack)) for schema, stack in self.get_iter()])
	def __copy__(self) -> _CompileTracker:
		return _CompileTracker([copy(stack) for stack in self._stacks])
	def __contains__(self, key: SchemaNode) -> bool:
		return key in self._stacks
	def __len__(self) -> int:
		return len(self._stacks)

@dataclass
class _CompileNode:
	parent_nodes: Set[SchemaNode]
	parent_ids: List[int]
	input_shape: LockedShape 
	priority: int
	def __copy__(self) -> _CompileNode:
		return _CompileNode(copy(self.parent_nodes), copy(self.parent_ids), self.input_shape, self.priority)

class _NodeCompileStack:
	NODE = 0
	PRIORITY = 1
	__slots__ = ["_stack", "_schema_node"]
	def __init__(self, schema_node: SchemaNode, stack: List[_CompileNode]) -> None:
		self._schema_node: SchemaNode = schema_node
		self._stack: List[_CompileNode] = stack
	def get_conformance(self, parent: SchemaNode, join_type: JoinType) -> Shape | None:
		if join_type != JoinType.NEW and (node := self.get_available(parent)) is not None:
			return self._schema_node.get_conformance_shape([node.input_shape])
		if join_type != JoinType.EXISTING:
			return OpenShape()
		return None
	def record(self, parent: SchemaNode, join_type: JoinType, input_shape: LockedShape, parent_id: int, priority: int) -> bool:
		if join_type != JoinType.NEW and (node := self.get_available(parent)) is not None:
			node.parent_nodes.add(parent)
			node.parent_ids.append(parent_id)
			node.input_shape = self._schema_node.get_mould_shape([node.input_shape, input_shape])
			return True
		if join_type != JoinType.EXISTING:
			self._stack.append(_CompileNode({parent}, [parent_id], input_shape, priority))
			return True
		return False
	def _get_available_index(self, parent: SchemaNode) -> int | None:
		for i, compile_node in enumerate(self._stack):
			if parent not in compile_node.parent_nodes:
				return i
		return None
	def get_available(self, parent: SchemaNode) -> _CompileNode | None:
		for compile_node in self._stack:
			if parent not in compile_node.parent_nodes:
				return compile_node 
		return None
	def pop(self) -> _CompileNode:
		return self._stack.pop()
	def peek(self) -> _CompileNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().priority if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)
	def __copy__(self) -> _NodeCompileStack:
		return _NodeCompileStack(self._schema_node, [copy(node) for node in self._stack])

