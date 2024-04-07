from __future__ import annotations

from ..shared import LockedShape, OpenShape, Shape, ID
from .schema_graph import SchemaNode, TransitionGroup, JoinType, Conformance, MAX_PRIORITY
from .compile_indices import CompilationIndices 
from .ir_node import IRNode

from dataclasses import dataclass

from copy import copy

class CompilationTracker:
	__slots__ = ["_stacks", "_stacks_lookup", "_id", "_max_id"]
	def __init__(self, stacks: list[NodeTrackerStack], stacks_lookup: dict[SchemaNode, int] | None, id: ID, max_id: ID) -> None:
		self._stacks: list[NodeTrackerStack] = stacks 
		self._stacks_lookup: dict[SchemaNode, int] = {}
		self._id: ID = id 
		self._max_id: ID = max_id 
		if stacks_lookup is not None:
			self._stacks_lookup = stacks_lookup
		else:
			self._stacks_lookup = {stack.get_schema(): i for i, stack in enumerate(stacks)}
	def compile_ir(self, indices: CompilationIndices) -> list[IRNode] | None:
		if self._id >= self._max_id:
			return None
		min_stack_index: int = min(range(len(self._stacks)), key=lambda i: self._stacks[i].get_priority())
		tracker_node = self._stacks[min_stack_index].pop()
		schema_node = self._stacks[min_stack_index].get_schema()
		index = indices.get_index(self._id, schema_node, tracker_node.input_shape)
		offset: int = int(index.get_shuffled(len(schema_node), 0))
		input_shape = schema_node.get_input_shape([tracker_node.input_shape])
		for group in (schema_node[(i + offset) % len(schema_node)] for i in range(len(schema_node))):
			if ((conformance := self.get_conformance(schema_node, group)) is not None
					and (output_shape := schema_node.get_output_shape(input_shape, conformance, index)) is not None
					and (ir := self.next(schema_node, group, output_shape).compile_ir(indices)) is not None):
				return ir + [IRNode(schema_node, tuple(tracker_node.parent_ids), self._id, input_shape, output_shape, index)]
		if (len(schema_node) == 0
				and (output_shape := schema_node.get_output_shape(input_shape, Conformance(OpenShape(), 1), index)) is not None):
			return [IRNode(schema_node, tuple(tracker_node.parent_ids), self._id, tracker_node.input_shape, output_shape, index)]
		return None
	def next(self, parent: SchemaNode, children: TransitionGroup, parent_output_shape: LockedShape) -> CompilationTracker:
		next_tracker = CompilationTracker(copy(self._stacks), copy(self._stacks_lookup), self._id + 1, self._max_id)
		for transition in children:
			stack_index = next_tracker._stacks_lookup[transition.get_next()]
			next_tracker._stacks[stack_index] = next_tracker._stacks[stack_index].next(parent, transition.get_join_type(), parent_output_shape, self._id, transition.get_priority())
		return next_tracker
	def get_conformance(self, parent: SchemaNode, transitions: TransitionGroup) -> Conformance | None:
		conformance: Conformance = Conformance(OpenShape(), 1) 
		for transition in transitions:
			if ((next_conformance := self[transition.get_next()].get_conformance(parent, transition.get_join_type())) is not None
					and (result := conformance.common(next_conformance)) is not None):
				conformance = result
			else:
				return None
		return conformance
	def stacks_str(self) -> str:
		return "\n".join([str(stack) for stack in self._stacks])
	def __getitem__(self, key: SchemaNode) -> NodeTrackerStack:
		if key in self._stacks_lookup:
			return self._stacks[self._stacks_lookup[key]]
		self._stacks.append(NodeTrackerStack(key, []))
		self._stacks_lookup[key] = len(self._stacks) - 1
		return self._stacks[-1]
	def __len__(self) -> int:
		return len(self._stacks)

@dataclass(frozen=True)
class NodeTracker:
	parent_nodes: set[SchemaNode]
	parent_ids: list[ID]
	input_shape: LockedShape 
	priority: int
	def copy_and_record(self, parent: SchemaNode, input_shape: LockedShape, parent_id: ID, priority: int) -> NodeTracker:
		return NodeTracker(self.parent_nodes | {parent}, self.parent_ids + [parent_id], input_shape, priority)

class NodeTrackerStack:
	__slots__ = ["_stack", "_schema_node"]
	def __init__(self, schema_node: SchemaNode, stack: list[NodeTracker]) -> None:
		self._schema_node: SchemaNode = schema_node
		self._stack: list[NodeTracker] = stack
	def get_conformance(self, parent: SchemaNode, join_type: JoinType) -> Conformance | None:
		if join_type != JoinType.NEW and (node_index := self._get_available_index(parent)) is not None:
			return self._schema_node.get_conformance([self._stack[node_index].input_shape])
		if join_type != JoinType.EXISTING:
			return self._schema_node.get_conformance([])
		return None
	def next(self, parent: SchemaNode, join_type: JoinType, parent_output_shape: LockedShape, parent_id: ID, priority: int) -> NodeTrackerStack:
		next_stack = NodeTrackerStack(self._schema_node, copy(self._stack))
		if join_type != JoinType.NEW and (node_index := self._get_available_index(parent)) is not None:
			next_stack._stack[node_index] = next_stack._stack[node_index].copy_and_record(parent,
				next_stack._schema_node.get_input_shape([next_stack._stack[node_index].input_shape, parent_output_shape]), parent_id, 
				min(priority, next_stack._stack[node_index].priority)) #min may not be correct
			return next_stack
		if join_type != JoinType.EXISTING:
			next_stack._stack.append(NodeTracker({parent}, [parent_id], parent_output_shape, priority))
			return next_stack
		raise ValueError("Join type not valid")
	def _get_available_index(self, parent: SchemaNode) -> int | None:
		for i in reversed(range(len(self._stack))):
			if parent not in self._stack[i].parent_nodes:
				return i
		return None
	def get_schema(self) -> SchemaNode:
		return self._schema_node
	def pop(self) -> NodeTracker:
		return self._stack.pop()
	def peek(self) -> NodeTracker:
		return self._stack[-1]
	def get_priority(self) -> int:
		return self.peek().priority if len(self._stack) > 0 else MAX_PRIORITY + 1
	def __len__(self) -> int:
		return len(self._stack)

