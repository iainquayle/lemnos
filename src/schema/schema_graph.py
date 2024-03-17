from __future__ import annotations

from ..shared import LockedShape, OpenShape, Shape, Index, ShapeBound
from .merge_method import MergeMethod 
from .activation import Activation
from .regularization import Regularization
from .transform import Transform  

from typing import Iterable
from typing_extensions import Self

from dataclasses import dataclass
from enum import Enum

from copy import copy

ID = int
@dataclass(frozen=True)
class ModuleIR:
	schema_node: SchemaNode
	parent_ids: tuple[ID]
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
	def __iter__(self) -> Iterable[TransitionGroup]:
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

class JoinType(Enum):
	EXISTING = "existing"
	NEW = "new"
	AUTO = "auto"

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
	def __getitem__(self, index: int) -> Transition:
		return self._transitions[index]
	def __iter__(self) -> Iterable[Transition]:
		return iter(self._transitions)
	def __len__(self) -> int:
		return len(self._transitions)
	def __str__(self) -> str:
		return f"TG{self._transitions}"
	def __repr__(self) -> str:
		return str(self)

class _CompileTracker:
	__slots__ = ["_stacks", "_stacks_lookup"]
	def __init__(self, stacks: list[_NodeCompileStack], stacks_lookup: dict[SchemaNode, int] | None) -> None:
		self._stacks: list[_NodeCompileStack] = stacks 
		self._stacks_lookup: dict[SchemaNode, int] = {}
		if stacks_lookup is not None:
			self._stacks_lookup = stacks_lookup
		else:
			self._stacks_lookup = {stack.get_schema(): i for i, stack in enumerate(stacks)}
	def compile_min(self) -> list[ModuleIR] | None:
		min_stack_index: int = min(range(len(self._stacks)), key=lambda i: self._stacks[i].get_priority())
		pass
	def is_empty(self) -> bool:
		for stack in self._stacks:
			if len(stack) > 0:
				return False
		return True
	def stacks_str(self) -> str:
		return "\n".join([str(stack) for stack in self._stacks])
	def __getitem__(self, key: SchemaNode) -> _NodeCompileStack:
		if key in self._stacks_lookup:
			return self._stacks[self._stacks_lookup[key]]
		self._stacks.append(_NodeCompileStack(key, []))
		self._stacks_lookup[key] = len(self._stacks) - 1
		return self._stacks[-1]
	def __copy__(self) -> _CompileTracker:
		return _CompileTracker([copy(stack) for stack in self._stacks], copy(self._stacks_lookup))
	def __len__(self) -> int:
		return len(self._stacks)

@dataclass
class _CompileNode:
	parent_nodes: Set[SchemaNode]
	parent_ids: list[int]
	input_shape: LockedShape 
	priority: int
	def __copy__(self) -> _CompileNode:
		return _CompileNode(copy(self.parent_nodes), copy(self.parent_ids), self.input_shape, self.priority)

class _NodeCompileStack:
	NODE = 0
	PRIORITY = 1
	__slots__ = ["_stack", "_schema_node"]
	def __init__(self, schema_node: SchemaNode, stack: list[_CompileNode]) -> None:
		self._schema_node: SchemaNode = schema_node
		self._stack: list[_CompileNode] = stack
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
	def get_schema(self) -> SchemaNode:
		return self._schema_node
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

