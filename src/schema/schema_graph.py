from __future__ import annotations

from ..shared import LockedShape, OpenShape, Shape, Index, ShapeBound
from .merge_method import MergeMethod 
from .activation import Activation
from .regularization import Regularization
from .transform import Transform  

from typing import Iterator 
from typing_extensions import Self

from dataclasses import dataclass
from enum import Enum
from abc import ABC as Abstract, abstractmethod

from copy import copy
from functools import reduce
import random

ID = int
@dataclass(frozen=True)
class ModuleIR:
	schema_node: SchemaNode
	parent_ids: tuple[ID, ...]
	transition_id: ID 
	input_shape: LockedShape
	output_shape: LockedShape
	index: Index

class SchemaNode:
	__slots__ = ["_transform", "_transition_groups", "_merge_method", "debug_name", "_activation", "_regularization", "_shape_bounds"]
	def __init__(self, 
			shape_bounds: ShapeBound,
			merge_method: MergeMethod,
			transform: Transform | None = None,
			activation: Activation | None = None,
			regularization: Regularization | None = None,
			debug_name: str = "") -> None:
		self._shape_bounds: ShapeBound = shape_bounds 
		self._transition_groups: list[TransitionGroup] = []
		self._merge_method: MergeMethod = merge_method 
		self._transform: Transform | None = transform 
		self._activation: Activation | None = activation 
		self._regularization: Regularization | None = regularization 
		self.debug_name: str = debug_name 
	def add_group(self, *transitions: tuple[SchemaNode, int, JoinType] | Transition) -> Self:
		self._transition_groups.append(TransitionGroup([transition if isinstance(transition, Transition) else Transition(*transition) for transition in transitions]))
		return self
	def get_mould_shape(self, input_shapes: list[LockedShape]) -> LockedShape:
		return self._merge_method.get_output_shape(input_shapes).squash(self.dimensionality())
	def get_output_shape(self, mould_shape: LockedShape, output_conformance: Shape, index: Index) -> LockedShape | None:
		output_shape = self._transform.get_output_shape(mould_shape, output_conformance, self._shape_bounds, index) if self._transform is not None else mould_shape
		if output_shape is not None and output_shape in self._shape_bounds and output_conformance.compatible(output_shape): 
			return output_shape 
		else:
			return None
	def get_conformance_shape(self, input_shapes: list[LockedShape]) -> Shape:
		return self._merge_method.get_conformance_shape(input_shapes)
	def get_transform(self) -> Transform | None:
		return self._transform
	def get_merge_method(self) -> MergeMethod:
		return self._merge_method
	def get_transition_groups(self) -> list[TransitionGroup]:
		return self._transition_groups
	def dimensionality(self) -> int:
		return len(self._shape_bounds)
	def __getitem__(self, index: int) -> TransitionGroup:
		return self._transition_groups[index]
	def __iter__(self) -> Iterator[TransitionGroup]:
		return iter(self._transition_groups)
	def get_inits_src(self, mould_shape: LockedShape, output_shape: LockedShape) -> list[str]:
		src: list[str] = []
		if self._transform is not None:
			src.append(self._transform.get_init_src(mould_shape, output_shape))
		if self._activation is not None:
			src.append(self._activation.get_init_src(mould_shape))
		if self._regularization is not None:
			src.append(self._regularization.get_init_src(mould_shape))
		return src
	def compile_IR(self, compile_tracker: _CompilationTracker, indices: BuildIndices, sequence_index: int) -> list[ModuleIR] | None: 
		node_info = compile_tracker[self].pop()
		index = Index()
		offset = index.get_shuffled(len(self._transition_groups), 0)
		for group in (self._transition_groups[(i + offset) % len(self._transition_groups)] for i in range(len(self._transition_groups))):
			if ((conformance := compile_tracker.get_conformance(self, group)) is not None
					and (output_shape := self.get_output_shape(node_info.input_shape, conformance, index)) is not None
					and (ir := compile_tracker.next(self, group, output_shape, node_info.priority).compile_IR(indices, sequence_index)) is not None):
				return ir + [ModuleIR(self, tuple(node_info.parent_ids), compile_tracker.get_id(), node_info.input_shape, output_shape, index)]
		if (len(self._transition_groups) == 0
				and (output_shape := self.get_output_shape(node_info.input_shape, OpenShape(), index)) is not None):
			return [ModuleIR(self, tuple(node_info.parent_ids), compile_tracker.get_id(), node_info.input_shape, output_shape, index)]
		return None
class JoinType(Enum):
	EXISTING = "existing"
	NEW = "new"
	AUTO = "auto"

#make transition and transition group dataclasses
#make immutable
class Transition:
	MAX_PRIORITY: int = 128 
	MIN_PRIORITY: int = 0 
	__slots__ = ["_next", "_optional", "_priority", "_join_type"]
	def __init__(self, next: SchemaNode, priority: int, join_type: JoinType = JoinType.NEW) -> None:
		if priority > Transition.MAX_PRIORITY or priority < Transition.MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self._next: SchemaNode = next
		self._optional: bool =  False 
		self._priority: int = priority 
		self._join_type: JoinType = join_type 
	def get_next(self) -> SchemaNode:
		return self._next
	def get_priority(self) -> int:
		return self._priority
	#def is_optional(self) -> bool:
	#	return self._optional
	def get_join_type(self) -> JoinType:
		return self._join_type
	def is_join_new(self) -> bool:
		return self._join_type == JoinType.NEW
	def is_join_existing(self) -> bool:
		return self._join_type == JoinType.EXISTING
	@staticmethod
	def get_max_priority() -> int:
		return Transition.MAX_PRIORITY 
	@staticmethod
	def get_min_priority() -> int:
		return Transition.MIN_PRIORITY


class TransitionGroup:
	__slots__ = ["_transitions"]
	def __init__(self, transitions: list[Transition]) -> None:
		self._transitions: list[Transition] = copy(transitions)
	def set_transitions(self, transitions: list[Transition]) -> None:
		pattern_set: set[SchemaNode] = set()
		for transition in transitions:
			if transition.get_next() in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.get_next())
		self._transitions = transitions
	def get_transitions(self) -> list[Transition]:
		return self._transitions
	def __iter__(self) -> Iterator[Transition]:
		return iter(self._transitions)
	def __len__(self) -> int:
		return len(self._transitions)

class _CompilationTracker:
	__slots__ = ["_stacks", "_stacks_lookup", "_id"]
	def __init__(self, stacks: list[_CompilationNodeStack], stacks_lookup: dict[SchemaNode, int] | None, id: ID) -> None:
		self._stacks: list[_CompilationNodeStack] = stacks 
		self._stacks_lookup: dict[SchemaNode, int] = {}
		self._id: ID = id 
		if stacks_lookup is not None:
			self._stacks_lookup = stacks_lookup
		else:
			self._stacks_lookup = {stack.get_schema(): i for i, stack in enumerate(stacks)}
	def compile_IR(self, indices: BuildIndices, sequence_index: int) -> list[ModuleIR] | None:
		min_stack_index: int = min(range(len(self._stacks)), key=lambda i: self._stacks[i].get_priority())
		return self._stacks[min_stack_index].get_schema().compile_IR(self, indices, sequence_index)
	def next(self, parent: SchemaNode, children: TransitionGroup, parent_output_shape: LockedShape, priority: int) -> _CompilationTracker:
		next_tracker = copy(self)
		for transition in children:
			stack_index = next_tracker._stacks_lookup[transition.get_next()]
			next_tracker._stacks[stack_index] = next_tracker._stacks[stack_index].next(parent, transition.get_join_type(), parent_output_shape, self.get_id(), priority)
		return next_tracker
	def get_conformance(self, parent: SchemaNode, children: TransitionGroup) -> Shape | None:
		common_conformance: Shape = OpenShape() 
		for transition in children:
			if ((conformance := self[transition.get_next()].get_conformance(parent, transition.get_join_type())) is not None
					and (conformance := common_conformance.common_lossless(conformance)) is not None):
				common_conformance = conformance
			else:
				return None
		return common_conformance
	def stacks_str(self) -> str:
		return "\n".join([str(stack) for stack in self._stacks])
	def __getitem__(self, key: SchemaNode) -> _CompilationNodeStack:
		if key in self._stacks_lookup:
			return self._stacks[self._stacks_lookup[key]]
		self._stacks.append(_CompilationNodeStack(key, []))
		self._stacks_lookup[key] = len(self._stacks) - 1
		return self._stacks[-1]
	def __copy__(self) -> _CompilationTracker:
		return _CompilationTracker(copy(self._stacks), copy(self._stacks_lookup), self._id + 1)
	def __len__(self) -> int:
		return len(self._stacks)
	def get_id(self) -> ID:
		return self._id

@dataclass(frozen=True)
class _CompilationNode:
	parent_nodes: set[SchemaNode]
	parent_ids: list[ID]
	input_shape: LockedShape 
	priority: int
	def copy_and_record(self, parent: SchemaNode, input_shape: LockedShape, parent_id: ID, priority: int) -> _CompilationNode:
		return _CompilationNode(self.parent_nodes | {parent}, self.parent_ids + [parent_id], input_shape, priority)

class _CompilationNodeStack:
	__slots__ = ["_stack", "_schema_node"]
	def __init__(self, schema_node: SchemaNode, stack: list[_CompilationNode]) -> None:
		self._schema_node: SchemaNode = schema_node
		self._stack: list[_CompilationNode] = stack
	def get_conformance(self, parent: SchemaNode, join_type: JoinType) -> Shape | None:
		if join_type != JoinType.NEW and (node_index := self._get_available_index(parent)) is not None:
			return self._schema_node.get_conformance_shape([self._stack[node_index].input_shape])
		if join_type != JoinType.EXISTING:
			return OpenShape()
		return None
	def next(self, parent: SchemaNode, join_type: JoinType, parent_output_shape: LockedShape, parent_id: ID, priority: int) -> _CompilationNodeStack:
		next_stack = _CompilationNodeStack(self._schema_node, copy(self._stack))
		if join_type != JoinType.NEW and (node_index := self._get_available_index(parent)) is not None:
			next_stack._stack[node_index] = next_stack._stack[node_index].copy_and_record(parent,
				next_stack._schema_node.get_mould_shape([next_stack._stack[node_index].input_shape, parent_output_shape]), parent_id, priority)
			return next_stack
		if join_type != JoinType.EXISTING:
			next_stack._stack.append(_CompilationNode({parent}, [parent_id], parent_output_shape, priority))
			return next_stack
		raise ValueError("Join type not valid")
	def _get_available_index(self, parent: SchemaNode) -> int | None:
		for i in reversed(range(len(self._stack))):
			if parent not in self._stack[i].parent_nodes:
				return i
		return None
	def get_schema(self) -> SchemaNode:
		return self._schema_node
	def pop(self) -> _CompilationNode:
		return self._stack.pop()
	def peek(self) -> _CompilationNode:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().priority if len(self._stack) > 0 else Transition.get_max_priority() + 1
	def __len__(self) -> int:
		return len(self._stack)

class BuildIndices(Abstract):
	@abstractmethod
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> tuple[Index, int]:	
		pass

class StaticIndices(BuildIndices):
	__slots__ = ["_indices"]
	def __init__(self, indices: list[Index]) -> None:
		self._indices: list[Index] = indices
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> tuple[Index, int]:
		return self._indices[id], 0 

class BreedIndices(BuildIndices):
	__slots__ = ["_sequences", "_sequence_change_prod", "_mutate_prod"]
	def __init__(self, sequence_change_prod: float = 0, mutate_prod: float = 0, sequences: list[list[tuple[Index, SchemaNode, LockedShape]]] = []) -> None:
		if sequence_change_prod < 0 or sequence_change_prod > 1 or mutate_prod < 0 or mutate_prod > 1:
			raise ValueError("Invalid probabilities")
		self._sequences: list[list[tuple[Index, SchemaNode, LockedShape]]] = sequences
		self._sequence_change_prod: float = sequence_change_prod
		self._mutate_prod: float = mutate_prod
	def get_index(self, id: int, sequence_index: int, schema_node: SchemaNode, shape_in: LockedShape) -> tuple[Index, int]:
		def search_sequence(sequence_index: int) -> tuple[Index, int] | None:
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
				sequence_indices: list[int] = list(range(sequence_index)) + list(range(sequence_index + 1, len(self._sequences)))
				random.shuffle(sequence_indices)
				for sequence in sequence_indices:
					if (result := search_sequence(sequence)) is not None:
						index, _ = result
						return index, sequence 
		return Index.random(), sequence_index 

