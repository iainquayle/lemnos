from __future__ import annotations

from ..shared import LockedShape, OpenShape, ShapeConformance, ShapeBound, ID
from .components.transformation import Transformation 
from .components.activation import Activation
from .components.regularization import Regularization
from .components.aggregation import Aggregation 
from .components.component import Component

import random
from copy import copy

from typing import Iterator, Iterable, Callable, Any
from typing_extensions import Self

from dataclasses import dataclass
from abc import ABC as Abstract, abstractmethod


@dataclass(frozen=True)
class IRNode:
	schema_node: SchemaNode
	parent_ids: tuple[ID, ...]
	id: ID 
	input_shape: LockedShape
	output_shape: LockedShape
	shape_trace: list[LockedShape]
	index: CompilationIndex

	def __str__(self) -> str:
		return f"SchemaNode: {self.schema_node.debug_name}, Parent IDs: {self.parent_ids}, ID: {self.id}, Input Shape: {self.input_shape}, Output Shape: {self.output_shape}, CompilationIndex: {self.index}"


# English version of how compilation works:
#
# The node with the lowest priority at the top of a stack is popped from the compilation tracker.
# A transition group is selected, and attempts to be added to the tracker, this involves:
#	- Creating new nodes for next schema nodes that are designated to be new, and set the priority to the priority of the taken transition.
#	- Attempting to join existing nodes, this involves:
#		- Get first available matching node, that has not been parented by the current node type before.
#		- Get the input shape (ShapeConformance) of the child node (ie all of the other parent nodes, after their shapes have been mergered according to the merge method of the child node).
#		- Attempt to merge the output shape of the current node with the conformance shape:
#			- If successful, update the child node in the tracker with the new parent, priority given by the transition and conformance shape, and create new IR node and return it in a list.
#			- If unsuccessful, but transition set to auto, create new node, else fail whole transition group.
# If all transition groups fail, attempt next transition group from previous node in the build stack.
# Repeat this until compilation tracker is empty, if a IR list is returned it is successful, else no solution was found.
#
# Note, this current implementation cannot solve all valid graphs, as it only has a single look ahead, and all previous compilation efforts in the stack are immutable.
#	However if the graph creator takes into account the single look ahead, it will always find a solution.


class SchemaNode:
	__slots__ = ["_transformation", "_transition_groups", "_growth_function", "_aggregation", "debug_name", "_activation", "_regularization", "_shape_bound"]

	def __init__(self, 
			shape_bound: ShapeBound | tuple[tuple[int | None, int | None], int | None],
			growth_function: Callable[[LockedShape, CompilationIndex], float] | None = None,
			aggregation: Aggregation | None = None,
			transformation: Transformation | None = None,
			activation: Activation | None = None,
			regularization: Regularization | None = None,
			debug_name: str = "") -> None:
		#this may not be smart as currently it may be needed as a shape guard
		#if aggregation is None and transformation is None and activation is None and regularization is None:
		#	raise ValueError("At least one component must be defined")
		self._shape_bound: ShapeBound = shape_bound if isinstance(shape_bound, ShapeBound) else ShapeBound(*shape_bound)
		self._growth_function: Callable[[LockedShape, CompilationIndex], float] | None = growth_function 
		self._transition_groups: list[TransitionGroup] = []
		self._aggregation: Aggregation | None = aggregation 
		self._transformation: Transformation | None = transformation 
		self._activation: Activation | None = activation 
		self._regularization: Regularization | None = regularization 
		self.debug_name: str = debug_name 

	def _compile(self, compilation_node: _CompilationNode, tracker: _CompilationTracker, indices: CompilationIndices, id: ID, max_id: ID) -> list[IRNode] | None:
		if id >= max_id:
			return None
		input_shape = self.get_input_shape([compilation_node.input_shape])
		index = indices.get_index(id, self, input_shape)
		offset = int(index.get_shuffled(len(self), 0))
		for group in (self[(i + offset) % len(self)] for i in range(len(self))):
			if ((conformance := group.get_conformance(tracker, self)) is not None
					and (output_shape := self.get_output_shape(input_shape, conformance, index)) is not None):
				next_tracker = group.join_nodes(tracker, self, output_shape, id)
				next_schema, next_node = next_tracker.pop_min()
				if (ir := next_schema._compile(next_node, next_tracker, indices, id + 1, max_id)) is not None:
					return ir + [IRNode(self, tuple(compilation_node.parent_ids), id, input_shape, output_shape, [], index)]
		if (len(self) == 0 and (output_shape := self.get_output_shape(input_shape, ShapeConformance(OpenShape(), 1), index)) is not None):
			return [IRNode(self, tuple(compilation_node.parent_ids), id, input_shape, output_shape, [], index)]
		return None

	def get_input_shape(self, input_shapes: list[LockedShape]) -> LockedShape:
		if self._aggregation is None:
			if len(input_shapes) > 1:
				raise ValueError("No merge method defined for multiple inputs")
			return input_shapes[0].squash(self.dimensionality())
		else:
			return self._aggregation.get_merged_shape(input_shapes).squash(self.dimensionality())

	def get_output_shape(self, input_shape: LockedShape, conformance: ShapeConformance, index: CompilationIndex) -> LockedShape | None:
		growth_factor = self._growth_function(input_shape, index) if self._growth_function is not None else 1
		bounds = self._shape_bound
		if self._activation is not None:
			conformance, bounds, growth_factor = self._activation.scale_build_conformances(conformance, bounds, growth_factor)
		output_shape = self._transformation.get_output_shape(input_shape, conformance, bounds, growth_factor) if self._transformation is not None else input_shape
		if output_shape is not None:
			output_shape = self._activation.scale_output_shape(output_shape) if self._activation is not None else output_shape
			if output_shape in self._shape_bound and conformance.is_compatible(output_shape): 
				return output_shape 
		return None

	def get_conformance(self, parent_shapes: list[LockedShape]) -> ShapeConformance | None:
		conformance_shape = OpenShape()
		if (self._aggregation is not None
					and (conformance_shape := self._aggregation.get_conformance_shape(parent_shapes)) is None):
			return None
		divisor = self._transformation.get_known_divisor() if self._transformation is not None else 1
		divisor = self._activation.scale_divisor(divisor) if self._activation is not None else divisor 
		return ShapeConformance(conformance_shape, divisor)

	def add_group(self, *transitions: Transition) -> Self:
		self._transition_groups.append(TransitionGroup(transitions))
		return self

	def get_aggregation(self) -> Aggregation | None:
		return self._aggregation

	def get_transformation(self) -> Transformation | None:
		return self._transformation

	def get_activation(self) -> Activation | None:
		return self._activation

	def get_regularization(self) -> Regularization | None:
		return self._regularization

	def get_components(self) -> list[Component]:
		return [component for component in (self._aggregation, self._transformation, self._activation, self._regularization) if component is not None]

	def dimensionality(self) -> int:
		return len(self._shape_bound)

	def __getitem__(self, index: int) -> TransitionGroup:
		return self._transition_groups[index]

	def __iter__(self) -> Iterator[TransitionGroup]:
		return iter(self._transition_groups)

	def __len__(self) -> int:
		return len(self._transition_groups)


class TransitionGroup:
	__slots__ = ["_transitions"]

	def __init__(self, transitions: Iterable[Transition]) -> None:
		pattern_set: set[SchemaNode] = set()
		for transition in transitions:
			if transition.get_next() in pattern_set:
				raise ValueError("Duplicate state in transition group")
			pattern_set.add(transition.get_next())
		self._transitions: tuple[Transition, ...] = tuple(transitions) 

	def get_conformance(self, tracker: _CompilationTracker, parent: SchemaNode) -> ShapeConformance | None:
		conformance: ShapeConformance = ShapeConformance(OpenShape(), 1)
		for transition in self._transitions:
			if ((next_conformance := transition.get_conformance(tracker, parent)) is not None
					and (next_conformance := conformance.common(next_conformance)) is not None):
				conformance = next_conformance 
			else:
				return None
		return conformance

	def join_nodes(self, tracker: _CompilationTracker, parent: SchemaNode, parent_shape: LockedShape, id: ID) -> _CompilationTracker:
		next_tracker = copy(tracker)
		for transition in self._transitions:
			transition.join_node(next_tracker, parent, parent_shape, id)
		return next_tracker

	def __iter__(self) -> Iterator[Transition]:
		return iter(self._transitions)

	def __len__(self) -> int:
		return len(self._transitions)


MAX_PRIORITY: int = 128 
MIN_PRIORITY: int = 0 
class Transition(Abstract):
	__slots__ = ["_next", "_priority", "_growth_function"]

	def __init__(self, next: SchemaNode, priority: int) -> None:
		if priority > MAX_PRIORITY or priority < MIN_PRIORITY:
			raise ValueError("Priority out of bounds")
		self._next: SchemaNode = next
		self._priority: int = priority 

	def get_next(self) -> SchemaNode:
		return self._next

	def get_priority(self) -> int:
		return self._priority
	@abstractmethod

	def get_conformance(self, tracker: _CompilationTracker, parent: SchemaNode) -> ShapeConformance | None:
		pass
	@abstractmethod

	def join_node(self, tracker: _CompilationTracker, parent: SchemaNode, parent_shape: LockedShape, parent_id: ID) -> _CompilationTracker:
		pass


class New(Transition):

	def get_conformance(self, tracker: _CompilationTracker, parent: SchemaNode) -> ShapeConformance | None:
		return self._next.get_conformance([])

	def join_node(self, tracker: _CompilationTracker, parent: SchemaNode, parent_shape: LockedShape, parent_id: ID) -> _CompilationTracker:
		tracker.get_mutable(self._next).push(_CompilationNode({parent}, [parent_id], parent_shape, self._priority))
		return tracker 


class Existing(Transition):

	def get_conformance(self, tracker: _CompilationTracker, parent: SchemaNode) -> ShapeConformance | None:
		if (compilation_node := tracker.get_immutable(self._next).get_immutable(parent)) is not None:
			return self._next.get_conformance([compilation_node.input_shape])
		return None

	def join_node(self, tracker: _CompilationTracker, parent: SchemaNode, parent_shape: LockedShape, parent_id: ID) -> _CompilationTracker:
		if (compilation_node := tracker.get_mutable(self._next).get_mutable(parent)) is not None:
			compilation_node.record(parent, parent_id, self._next.get_input_shape([compilation_node.input_shape, parent_shape]), self._priority)
		return tracker


class Auto(Transition):

	def get_conformance(self, tracker: _CompilationTracker, parent: SchemaNode) -> ShapeConformance | None:
		if (compilation_node := tracker.get_immutable(self._next).get_immutable(parent)) is not None:
			return self._next.get_conformance([compilation_node.input_shape])
		return self._next.get_conformance([])

	def join_node(self, tracker: _CompilationTracker, parent: SchemaNode, parent_shape: LockedShape, parent_id: ID) -> _CompilationTracker:
		stack = tracker.get_mutable(self._next)
		if (compilation_node := stack.get_mutable(parent)) is not None:
			compilation_node.record(parent, parent_id, self._next.get_input_shape([compilation_node.input_shape, parent_shape]), self._priority)
		stack.push(_CompilationNode({parent}, [parent_id], parent_shape, self._priority))
		return tracker


class _CompilationTracker:
	__slots__ = ["_stacks", "_stacks_lookup", "_id", "_max_id"]

	def __init__(self, stacks: list[_CompilationNodeStack], stacks_lookup: dict[SchemaNode, int] | None) -> None:
		self._stacks: list[_CompilationNodeStack] = stacks 
		self._stacks_lookup: dict[SchemaNode, int] = {}
		if stacks_lookup is not None:
			self._stacks_lookup = stacks_lookup
		else:
			self._stacks_lookup = {stack.get_schema(): i for i, stack in enumerate(stacks)}

	def pop_min(self) -> tuple[SchemaNode, _CompilationNode]: 
		min_stack_index: int = min(range(len(self._stacks)), key=lambda i: self._stacks[i].get_priority())
		if len(self._stacks[min_stack_index]) == 0:
			raise ValueError("Empty stack")
		self._stacks[min_stack_index] = copy(self._stacks[min_stack_index])
		return self._stacks[min_stack_index].get_schema(), self._stacks[min_stack_index].pop()

	def stacks_str(self) -> str:
		return "\n".join([str(stack) for stack in self._stacks])

	def get_mutable(self, node: SchemaNode) -> _CompilationNodeStack:
		if node in self._stacks_lookup:
			self._stacks[self._stacks_lookup[node]] = copy(self._stacks[self._stacks_lookup[node]])
			return self._stacks[self._stacks_lookup[node]]
		self._stacks.append(_CompilationNodeStack(node, []))
		self._stacks_lookup[node] = len(self._stacks) - 1
		return self._stacks[-1]

	def get_immutable(self, node: SchemaNode) -> _CompilationNodeStack:
		#these are not actually immutable, just dont break the trust
		#safest would be to copy every time
		if node in self._stacks_lookup:
			return self._stacks[self._stacks_lookup[node]]
		return _CompilationNodeStack(node, [])

	def __len__(self) -> int:
		return len(self._stacks)

	def __copy__(self) -> _CompilationTracker:
		return _CompilationTracker(copy(self._stacks), copy(self._stacks_lookup))


class _CompilationNodeStack:
	__slots__ = ["_stack", "_schema_node"]

	def __init__(self, schema_node: SchemaNode, stack: list[_CompilationNode]) -> None:
		self._schema_node: SchemaNode = schema_node
		self._stack: list[_CompilationNode] = stack

	def get_mutable(self, parent: SchemaNode) -> _CompilationNode | None:
		if (node_index := self._get_available_index(parent)) is not None:
			self._stack[node_index] = copy(self._stack[node_index])
			return self._stack[node_index]
		return None

	def get_immutable(self, parent: SchemaNode) -> _CompilationNode | None:
		if (node_index := self._get_available_index(parent)) is not None:
			return self._stack[node_index]
		return None

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

	def push(self, node: _CompilationNode) -> None:
		self._stack.append(node)

	def get_priority(self) -> int:
		return self.peek().priority if len(self._stack) > 0 else MAX_PRIORITY + 1

	def __len__(self) -> int:
		return len(self._stack)

	def __copy__(self) -> _CompilationNodeStack:
		return _CompilationNodeStack(self._schema_node, copy(self._stack))


@dataclass(frozen=False)
class _CompilationNode:
	parent_nodes: set[SchemaNode]
	parent_ids: list[ID]
	input_shape: LockedShape 
	priority: int

	def record(self, parent: SchemaNode, parent_id: ID, new_input_shape: LockedShape, priority: int) -> None:
		self.parent_nodes.add(parent)
		self.parent_ids.append(parent_id)
		self.input_shape = new_input_shape
		self.priority = priority

	def __copy__(self) -> _CompilationNode:
		return _CompilationNode(copy(self.parent_nodes), copy(self.parent_ids), self.input_shape, self.priority)


class CompilationIndices(Abstract):
	@abstractmethod

	def get_index(self, id: ID, schema_node: SchemaNode, shape_in: LockedShape) -> CompilationIndex:	
		pass


class CompilationIndex:
	__slots__ = ["_index"]

	def __init__(self, index: int = 0) -> None:
		self._index: int = index
	@staticmethod

	def random() -> CompilationIndex:
		return CompilationIndex(random.randint(0, 2**31 - 1))

	def get_shuffled(self, bounds: tuple[float, float] | float, salt: int = 0) -> float:
		if isinstance(bounds, float) or isinstance(bounds, int):
			bounds = (0, bounds)
		elif bounds[0] > bounds[1]:
			bounds = (bounds[1], bounds[0])
		return random.Random(self._index + salt).uniform(*bounds)

	def get(self) -> int:
		return self._index

	def __eq__(self, other: Any) -> bool:
		return isinstance(other, CompilationIndex) and self._index == other._index

	def __str__(self) -> str:
		return str(self._index)

	def __repr__(self) -> str:
		return f"CompilationIndex({self._index})"
